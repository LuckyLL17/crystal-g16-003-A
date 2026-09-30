import { useEffect, useMemo, useRef, useState, type ComponentRef } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { Grid, Html, Line, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

type DefectType = 'vacancy' | 'substitution' | 'dislocation';
type AxisId = 'x' | 'y' | 'z';
type SliceMode = AxisId | 'custom';
type AtomPoint = { key: string; x: number; y: number; z: number; defect?: 'vacancy' | 'substitution' | 'normal' };

const colors = { host: '#64c9e1', substitution: '#efb863', vacancy: '#ef7e83', bond: '#4b8190' };
const sliceColors = {
  face: '#5cc4d6',
  outline: '#b6f1f8',
  hostCap: '#8fe3ee',
  subCap: '#ffcf7e',
  bondPoint: '#d9faff',
  vacancyRing: '#ef7e83',
  dislocationPoint: '#f08d87',
  arrow: '#bff2f8',
};

const SPACING = 1.25;
const HOST_RADIUS = 0.2;
const SUB_RADIUS = 0.3;
const VACANCY_GHOST_RADIUS = 0.35;
const DEFAULT_CAMERA_POSITION: [number, number, number] = [8, 6, 8];
const DEFAULT_TARGET = new THREE.Vector3();
const SLICE_CAMERA_DISTANCE = 11;
const GROUP_ROTATION: [number, number, number] = [0.15, -0.35, 0];
const LATTICE_QUATERNION = new THREE.Quaternion().setFromEuler(new THREE.Euler(...GROUP_ROTATION));
const AXIS_NORMAL: Record<AxisId, THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};
// 刃型位错参考折线（与原场景一致，位于晶格局部空间）
const DISLOCATION_LINE: THREE.Vector3[] = [
  new THREE.Vector3(-1.9, -1.9, 0),
  new THREE.Vector3(-0.65, -1.9, 0),
  new THREE.Vector3(-0.65, -0.65, 0),
  new THREE.Vector3(0.65, -0.65, 0),
  new THREE.Vector3(0.65, 0.65, 0),
  new THREE.Vector3(1.9, 0.65, 0),
];

// ---------------------------------------------------------------------------
// 晶格数据（保持原演示数据逻辑）
// ---------------------------------------------------------------------------
function buildAtoms(cells: number, defect: DefectType): AtomPoint[] {
  const atoms: AtomPoint[] = [];
  for (let x = 0; x < cells; x += 1) for (let y = 0; y < cells; y += 1) for (let z = 0; z < cells; z += 1) {
    const key = `${x}-${y}-${z}`;
    if (defect === 'vacancy' && x === 1 && y === 1 && z === 1) continue;
    if (defect === 'substitution' && x === 1 && y === 1 && z === 1) atoms.push({ key, x, y, z, defect: 'substitution' });
    else atoms.push({ key, x, y, z, defect: 'normal' });
  }
  if (defect === 'dislocation') return atoms.filter(atom => !(atom.x === 2 && atom.y === 0 && atom.z === 2));
  return atoms;
}

const latticeToWorld = (cells: number) => {
  const half = ((cells - 1) / 2) * SPACING;
  return (x: number, y: number, z: number) =>
    new THREE.Vector3(x * SPACING - half, y * SPACING - half, z * SPACING - half);
};

function buildBonds(atoms: AtomPoint[], cells: number): [THREE.Vector3, THREE.Vector3][] {
  const toWorld = latticeToWorld(cells);
  const list: [THREE.Vector3, THREE.Vector3][] = [];
  atoms.forEach(atom => {
    const origin = toWorld(atom.x, atom.y, atom.z);
    const xNeighbor = atoms.find(item => item.x === atom.x + 1 && item.y === atom.y && item.z === atom.z);
    if (xNeighbor) list.push([origin, toWorld(xNeighbor.x, xNeighbor.y, xNeighbor.z)]);
    const zNeighbor = atoms.find(item => item.x === atom.x && item.y === atom.y && item.z === atom.z + 1);
    if (zNeighbor) list.push([origin, toWorld(zNeighbor.x, zNeighbor.y, zNeighbor.z)]);
  });
  return list;
}

// 结构范围：最外层原子球心再计入缺陷原子半径
function structureHalfExtent(cells: number): number {
  return ((cells - 1) / 2) * SPACING + SUB_RADIUS;
}

// ----------------------------------------------------------------
// 切片几何：真实截面（平面裁剪结构包围盒 + 被截原子圆截面 + 键/缺陷截点）
// ----------------------------------------------------------------
type AtomCut = { key: string; center: THREE.Vector3; radius: number; kind: 'host' | 'substitution' };
type BondCut = { point: THREE.Vector3 };
type BondSegment = { a: THREE.Vector3; b: THREE.Vector3 };
type SliceData = {
  normal: THREE.Vector3;
  offset: number;
  vertices: THREE.Vector3[];
  centroid: THREE.Vector3;
  basisU: THREE.Vector3;
  basisV: THREE.Vector3;
  quaternion: THREE.Quaternion;
  atomCuts: AtomCut[];
  bondCuts: BondCut[];
  bondSegments: BondSegment[];
  vacancyCenter: THREE.Vector3 | null;
  vacancyRadius: number;
  dislocationCuts: THREE.Vector3[];
  dislocationCoplanar: [THREE.Vector3, THREE.Vector3][];
  anchor: THREE.Vector3;
};

const EPS = 1e-6;

function segmentPlaneCross(a: THREE.Vector3, b: THREE.Vector3, n: THREE.Vector3, d: number): THREE.Vector3 | null {
  const sa = a.dot(n) - d;
  const sb = b.dot(n) - d;
  if (sa * sb < -EPS) {
    const t = sa / (sa - sb);
    return a.clone().lerp(b, t);
  }
  if (Math.abs(sa) <= EPS) return a.clone();
  if (Math.abs(sb) <= EPS) return b.clone();
  return null;
}

// 平面 n·r = d 与中心在原点、半边长 half 的 AABB 求交，返回按绕序排列的凸多边形
function intersectPlaneWithBox(n: THREE.Vector3, d: number, half: number): THREE.Vector3[] {
  const corners: THREE.Vector3[] = [];
  for (let i = 0; i < 8; i += 1) {
    corners.push(new THREE.Vector3(i & 1 ? half : -half, i & 2 ? half : -half, i & 4 ? half : -half));
  }
  const edges: [number, number][] = [
    [0, 1], [2, 3], [4, 5], [6, 7],
    [0, 2], [1, 3], [4, 6], [5, 7],
    [0, 4], [1, 5], [2, 6], [3, 7],
  ];
  const points: THREE.Vector3[] = [];
  for (const [ia, ib] of edges) {
    const p = segmentPlaneCross(corners[ia], corners[ib], n, d);
    if (p && !points.some(q => q.distanceToSquared(p) < 1e-8)) points.push(p);
  }
  if (points.length < 3) return points;

  const centroid = points.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / points.length);
  const reference = Math.abs(n.y) > 0.99 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const u = new THREE.Vector3().crossVectors(reference, n).normalize();
  const v = new THREE.Vector3().crossVectors(n, u).normalize();
  points.sort((a, b) => {
    const aa = Math.atan2(a.clone().sub(centroid).dot(v), a.clone().sub(centroid).dot(u));
    const bb = Math.atan2(b.clone().sub(centroid).dot(v), b.clone().sub(centroid).dot(u));
    return aa - bb;
  });
  return points;
}

function pointInsideConvexPolygon(p: THREE.Vector3, vertices: THREE.Vector3[], n: THREE.Vector3): boolean {
  for (let i = 0; i < vertices.length; i += 1) {
    const edge = vertices[(i + 1) % vertices.length].clone().sub(vertices[i]);
    const toPoint = p.clone().sub(vertices[i]);
    if(new THREE.Vector3().crossVectors(edge, toPoint).dot(n) < -1e-4) return false;
  }
  return true;
}

type PolylineIntersection = {
  crossings: THREE.Vector3[];
  coplanarSegments: [THREE.Vector3, THREE.Vector3][];
};

function polylinePlaneIntersections(points: THREE.Vector3[], n: THREE.Vector3, d: number): PolylineIntersection {
  const crossings: THREE.Vector3[] = [];
  const coplanarSegments: [THREE.Vector3, THREE.Vector3][] = [];
  for (let i = 0; i < points.length - 1; i += 1) {
    const a = points[i];
    const b = points[i + 1];
    const sa = a.dot(n) - d;
    const sb = b.dot(n) - d;
    // 整段缺陷线落在切片平面内：作为截面线段保留
    if (Math.abs(sa) <= EPS && Math.abs(sb) <= EPS) {
      coplanarSegments.push([a, b]);
      continue;
    }
    if (sa * sb < -EPS) {
      const hit = a.clone().lerp(b, sa / (sa - sb));
      // 顶点恰在平面上时相邻两段给出同一点，按位置去重
      if (!crossings.some(q => q.distanceToSquared(hit) < 1e-8)) crossings.push(hit);
    } else if (Math.abs(sa) <= EPS) {
      if (!crossings.some(q => q.distanceToSquared(a) < 1e-8)) crossings.push(a);
    } else if (Math.abs(sb) <= EPS) {
      if (!crossings.some(q => q.distanceToSquared(b) < 1e-8)) crossings.push(b);
    }
  }
  return { crossings, coplanarSegments };
}

function computeSlice(cells: number, defect: DefectType, normal: THREE.Vector3, offset: number): SliceData | null {
  const half = structureHalfExtent(cells);
  // 明显超出结构范围：直接返回空，不生成任何几何
  if (Math.abs(offset) > half + EPS) return null;

  const vertices = intersectPlaneWithBox(normal, offset, half);
  if (vertices.length < 3) return null;

  const reference = Math.abs(normal.y) > 0.99 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const basisU = new THREE.Vector3().crossVectors(reference, normal).normalize();
  const basisV = new THREE.Vector3().crossVectors(normal, basisU).normalize();
  const quaternion = new THREE.Quaternion().setFromRotationMatrix(new THREE.Matrix4().makeBasis(basisU, basisV, normal));
  const centroid = vertices.reduce((acc, p) => acc.add(p), new THREE.Vector3()).multiplyScalar(1 / vertices.length);

  const toWorld = latticeToWorld(cells);
  const atoms = buildAtoms(cells, defect);
  // 被截原子：球心到平面距离小于半径，截面圆半径 sqrt(R² - s²)
  const atomCuts: AtomCut[] = [];
  let substitutionCut: THREE.Vector3 | null = null;
  for (const atom of atoms) {
    const center = toWorld(atom.x, atom.y, atom.z);
    const radius = atom.defect === 'substitution' ? SUB_RADIUS : HOST_RADIUS;
    const distance = center.dot(normal) - offset;
    if (Math.abs(distance) < radius - EPS) {
      const cutCenter = center.clone().addScaledVector(normal, -distance);
      const cutRadius = Math.sqrt(Math.max(radius * radius - distance * distance, 0));
      const kind = atom.defect === 'substitution' ? 'substitution' : 'host';
      atomCuts.push({ key: atom.key, center: cutCenter, radius: cutRadius, kind });
      if (kind === 'substitution') substitutionCut = cutCenter;
    }
  }

  // 被截最近邻键：横穿的键记录截点，整段落在平面内的记录线段
  const bondCuts: BondCut[] = [];
  const bondSegments: BondSegment[] = [];
  for (const [a, b] of buildBonds(atoms, cells)) {
    const sa = a.dot(normal) - offset;
    const sb = b.dot(normal) - offset;
    if (Math.abs(sa) <= EPS && Math.abs(sb) <= EPS) {
      bondSegments.push({ a, b });
    } else if (sa * sb < -EPS) {
      const hit = a.clone().lerp(b, sa / (sa - sb));
      // 端点在平面上时相邻键会给出同一点，按位置去重
      if (!bondCuts.some(cut => cut.point.distanceToSquared(hit) < 1e-8)) bondCuts.push({ point: hit });
    }
  }

  // 空位标记：原空位位置的线框球与平面相交成环
  let vacancyCenter: THREE.Vector3 | null = null;
  let vacancyRadius = 0;
  if (defect === 'vacancy') {
    const site = toWorld(1, 1, 1);
    const distance = site.dot(normal) - offset;
    if (Math.abs(distance) < VACANCY_GHOST_RADIUS - EPS) {
      vacancyCenter = site.clone().addScaledVector(normal, -distance);
      vacancyRadius = Math.sqrt(Math.max(VACANCY_GHOST_RADIUS * VACANCY_GHOST_RADIUS - distance * distance, 0));
    }
  }

  // 刃型位错折线：横穿截点 + 共面线段（如沿 z=0 切片时整条折线落在截面内）
  const { crossings: dislocationCuts, coplanarSegments: dislocationCoplanar } =
    polylinePlaneIntersections(DISLOCATION_LINE, normal, offset);

  // 平面虽在包围盒内，但没有截到任何原子/键/缺陷元素（如层间间隙、与包围盒表面相切）：
  // 视为空态，不渲染误导性的裸多边形
  const intersectsStructure =
    atomCuts.length > 0 ||
    bondCuts.length > 0 ||
    bondSegments.length > 0 ||
    vacancyCenter !== null ||
    dislocationCuts.length > 0 ||
    dislocationCoplanar.length > 0;
  if (!intersectsStructure) return null;

  // 相机对焦：优先让缺陷本身落在视野中心，否则回退到截面质心
  const defectSite = defect === 'dislocation' ? toWorld(2, 0, 2) : toWorld(1, 1, 1);
  const projected = defectSite.clone().addScaledVector(normal, -(defectSite.dot(normal) - offset));
  const candidates = [vacancyCenter, substitutionCut, ...dislocationCuts, projected].filter(
    (p): p is THREE.Vector3 => p !== null && pointInsideConvexPolygon(p, vertices, normal),
  );
  const anchor = candidates[0] ?? centroid;

  return {
    normal,
    offset,
    vertices,
    centroid,
    basisU,
    basisV,
    quaternion,
    atomCuts,
    bondCuts,
    bondSegments,
    vacancyCenter,
    vacancyRadius,
    dislocationCuts,
    dislocationCoplanar,
    anchor,
  };
}

function resolveNormal(mode: SliceMode, custom: Record<AxisId, number>): THREE.Vector3 | null {
  if (mode !== 'custom') return AXIS_NORMAL[mode].clone();
  const v = new THREE.Vector3(custom.x, custom.y, custom.z);
  if (v.lengthSq() < EPS) return null;
  return v.normalize();
}

const formatComponent = (n: number) => (Math.abs(n) < 1e-8 ? 0 : Number(n.toFixed(2)));
const formatVector = (n: THREE.Vector3) => `(${formatComponent(n.x)}, ${formatComponent(n.y)}, ${formatComponent(n.z)})`;

// ----------------------------------------------------------------
// 场景内组件
// ----------------------------------------------------------------
function SliceGeometry({ data, opacity }: { data: SliceData; opacity: number }) {
  const faceGeometry = useMemo(() => {
    const shape = new THREE.Shape();
    data.vertices.forEach((p, i) => {
      const u = p.dot(data.basisU);
      const v = p.dot(data.basisV);
      if (i === 0) shape.moveTo(u, v);
      else shape.lineTo(u, v);
    });
    shape.closePath();
    return new THREE.ShapeGeometry(shape);
  }, [data]);

  const facePosition = useMemo(() => data.centroid.clone().addScaledVector(data.normal, 0.012), [data]);
  const outlinePoints = useMemo(() => [...data.vertices, data.vertices[0]], [data]);
  const arrowBase = useMemo(() => data.centroid.clone().addScaledVector(data.normal, 0.1), [data]);

  return (
    <group>
      {/* 截面衬底多边形 */}
      <mesh geometry={faceGeometry} position={facePosition} quaternion={data.quaternion} renderOrder={1}>
        <meshBasicMaterial
          color={sliceColors.face}
          transparent
          opacity={opacity}
          side={THREE.DoubleSide}
          depthWrite={false}
          polygonOffset
          polygonOffsetFactor={-1}
        />
      </mesh>
      <Line points={outlinePoints} color={sliceColors.outline} lineWidth={1.4} transparent opacity={0.9} />

      {/* 被截原子圆截面 */}
      {data.atomCuts.map(cut => (
        <mesh
          key={cut.key}
          position={cut.center.clone().addScaledVector(data.normal, 0.02)}
          quaternion={data.quaternion}
          renderOrder={2}
        >
          <circleGeometry args={[cut.radius, 30]} />
          <meshBasicMaterial
            color={cut.kind === 'substitution' ? sliceColors.subCap : sliceColors.hostCap}
            transparent
            opacity={Math.min(opacity + 0.35, 0.95)}
            side={THREE.DoubleSide}
            depthWrite={false}
          />
        </mesh>
      ))}

      {/* 键截点 */}
      {data.bondCuts.map((cut, i) => (
        <mesh
          key={`bond-${i}`}
          position={cut.point.clone().addScaledVector(data.normal, 0.025)}
          quaternion={data.quaternion}
          renderOrder={3}
        >
          <circleGeometry args={[0.05, 16]} />
          <meshBasicMaterial color={sliceColors.bondPoint} side={THREE.DoubleSide} transparent opacity={0.95} />
        </mesh>
      ))}

      {/* 共面键段：整层键落在截面内（如 cells=3 沿中间层切片） */}
      {data.bondSegments.map((seg, i) => (
        <Line
          key={`bond-seg-${i}`}
          points={[
            seg.a.clone().addScaledVector(data.normal, 0.02),
            seg.b.clone().addScaledVector(data.normal, 0.02),
          ]}
          color={sliceColors.bondPoint}
          lineWidth={1.6}
          transparent
          opacity={0.85}
        />
      ))}

      {/* 空位截面环 */}
      {data.vacancyCenter && (
        <mesh
          position={data.vacancyCenter.clone().addScaledVector(data.normal, 0.025)}
          quaternion={data.quaternion}
          renderOrder={3}
        >
          <ringGeometry args={[data.vacancyRadius * 0.55, data.vacancyRadius, 40]} />
          <meshBasicMaterial color={sliceColors.vacancyRing} side={THREE.DoubleSide} transparent opacity={0.9} />
        </mesh>
      )}

      {/* 刃型位错截点 */}
      {data.dislocationCuts.map((p, i) => (
        <mesh key={`dislocation-${i}`} position={p.clone().addScaledVector(data.normal, 0.04)} renderOrder={3}>
          <octahedronGeometry args={[0.09, 0]} />
          <meshBasicMaterial color={sliceColors.dislocationPoint} />
        </mesh>
      ))}

      {/* 共面缺陷线段：整条缺陷线落在截面内（如刃型位错沿 z=0 切片） */}
      {data.dislocationCoplanar.map((seg, i) => (
        <Line
          key={`dislocation-seg-${i}`}
          points={[
            seg[0].clone().addScaledVector(data.normal, 0.035),
            seg[1].clone().addScaledVector(data.normal, 0.035),
          ]}
          color={sliceColors.dislocationPoint}
          lineWidth={3.5}
        />
      ))}

      <NormalArrow base={arrowBase} normal={data.normal} label={formatVector(data.normal)} />
    </group>
  );
}

function NormalArrow({ base, normal, label }: { base: THREE.Vector3; normal: THREE.Vector3; label: string }) {
  const length = 1.6;
  const quaternion = useMemo(
    () => new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), normal),
    [normal],
  );
  const shaftPosition = useMemo(() => base.clone().addScaledVector(normal, length * 0.41), [base, normal]);
  const tipPosition = useMemo(() => base.clone().addScaledVector(normal, length), [base, normal]);
  const labelPosition = useMemo(() => tipPosition.clone().addScaledVector(normal, 0.34), [tipPosition, normal]);

  return (
    <group renderOrder={4}>
      <mesh position={shaftPosition} quaternion={quaternion}>
        <cylinderGeometry args={[0.018, 0.018, length * 0.82, 10]} />
        <meshBasicMaterial color={sliceColors.arrow} />
      </mesh>
      <mesh position={tipPosition} quaternion={quaternion}>
        <coneGeometry args={[0.09, 0.28, 16]} />
        <meshBasicMaterial color={sliceColors.arrow} />
      </mesh>
      <Html position={labelPosition} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
        <div className="plane-normal-tag">
          <b>n̂</b>
          <span>{label}</span>
        </div>
      </Html>
    </group>
  );
}

// 相机沿法向定位；run-token 保证连续切换时旧动画立即作废，关闭时恢复原视图
function SliceCameraRig({
  controlsRef,
  enabled,
  normalKey,
  flyNonce,
  worldNormal,
  worldTarget,
  valid,
}: {
  controlsRef: React.RefObject<ComponentRef<typeof OrbitControls> | null>;
  enabled: boolean;
  normalKey: string;
  flyNonce: number;
  worldNormal: THREE.Vector3 | null;
  worldTarget: THREE.Vector3 | null;
  valid: boolean;
}) {
  const camera = useThree(state => state.camera);
  const latest = useRef({ worldNormal, worldTarget, valid });
  latest.current = { worldNormal, worldTarget, valid };
  const savedView = useRef<{ position: THREE.Vector3; target: THREE.Vector3 } | null>(null);
  const runToken = useRef(0);

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;

    let raf = 0;
    const flyTo = (endPosition: THREE.Vector3, endTarget: THREE.Vector3, token: number) => {
      const startPosition = camera.position.clone();
      const startTarget = controls.target.clone();
      const startedAt = performance.now();
      const duration = 650;
      const step = (now: number) => {
        if (runToken.current !== token) return; // 已被更新的切换作废
        const t = Math.min(1, (now - startedAt) / duration);
        const eased = 1 - (1 - t) ** 3;
        camera.position.lerpVectors(startPosition, endPosition, eased);
        controls.target.lerpVectors(startTarget, endTarget, eased);
        controls.update();
        if (t < 1) raf = requestAnimationFrame(step);
      };
      raf = requestAnimationFrame(step);
    };

    if (!enabled) {
      const saved = savedView.current;
      if (saved) {
        runToken.current += 1;
        flyTo(saved.position, saved.target, runToken.current);
        savedView.current = null;
      }
      return () => {
        cancelAnimationFrame(raf);
        runToken.current += 1;
      };
    }

    const current = latest.current;
    // 空态切片不移动相机，保持当前观察角度
    if (!current.valid || !current.worldNormal || !current.worldTarget) return;

    if (!savedView.current) {
      savedView.current = { position: camera.position.clone(), target: controls.target.clone() };
    }
    runToken.current += 1;
    const endPosition = current.worldTarget.clone().addScaledVector(current.worldNormal, SLICE_CAMERA_DISTANCE);
    flyTo(endPosition, current.worldTarget, runToken.current);

    return () => {
      cancelAnimationFrame(raf);
      runToken.current += 1;
    };
    // 仅在开关、法向切换或主动重置时飞行动画；偏移/透明度变化不重新定位
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [enabled, normalKey, flyNonce]);

  return null;
}

function DefectScene({
  cells,
  defect,
  showBonds,
  planeOpacity,
  sliceData,
  sliceOpacity,
  sliceEnabled,
  sliceNormalKey,
  flyNonce,
  worldNormal,
  worldTarget,
}: {
  cells: number;
  defect: DefectType;
  showBonds: boolean;
  planeOpacity: number;
  sliceData: SliceData | null;
  sliceOpacity: number;
  sliceEnabled: boolean;
  sliceNormalKey: string;
  flyNonce: number;
  worldNormal: THREE.Vector3 | null;
  worldTarget: THREE.Vector3 | null;
}) {
  const controlsRef = useRef<ComponentRef<typeof OrbitControls> | null>(null);
  const atoms = useMemo(() => buildAtoms(cells, defect), [cells, defect]);
  const bonds = useMemo(() => buildBonds(atoms, cells), [atoms, cells]);
  const toWorld = useMemo(() => latticeToWorld(cells), [cells]);
  const vacancySite = useMemo(() => toWorld(1, 1, 1), [toWorld]);

  return (
    <Canvas shadows camera={{ position: DEFAULT_CAMERA_POSITION, fov: 38 }}>
      <color attach="background" args={['#09151d']} />
      <ambientLight intensity={1.25} />
      <directionalLight position={[5, 8, 5]} intensity={2.3} castShadow />
      <pointLight position={[-4, 2, 3]} color="#5fd4ee" intensity={12} distance={12} />

      <group rotation={GROUP_ROTATION}>
        {showBonds &&
          bonds.map((bond, index) => (
            <Line key={index} points={bond} color={colors.bond} lineWidth={1.2} transparent opacity={0.65} />
          ))}

        {atoms.map(atom => {
          const position: [number, number, number] = [
            (atom.x - (cells - 1) / 2) * SPACING,
            (atom.y - (cells - 1) / 2) * SPACING,
            (atom.z - (cells - 1) / 2) * SPACING,
          ];
          const color = atom.defect === 'substitution' ? colors.substitution : colors.host;
          return (
            <mesh key={atom.key} position={position} castShadow>
              <sphereGeometry args={[atom.defect === 'substitution' ? 0.3 : 0.2, 18, 12]} />
              <meshStandardMaterial
                color={color}
                emissive={atom.defect === 'substitution' ? colors.substitution : '#000'}
                emissiveIntensity={atom.defect === 'substitution' ? 0.35 : 0}
              />
            </mesh>
          );
        })}

        {defect === 'vacancy' && (
          <mesh position={vacancySite}>
            <sphereGeometry args={[VACANCY_GHOST_RADIUS, 16, 12]} />
            <meshBasicMaterial color={colors.vacancy} transparent opacity={0.26} wireframe />
          </mesh>
        )}
        {defect === 'dislocation' && <Line points={DISLOCATION_LINE} color="#f08d87" lineWidth={4} />}

        {/* 真实截面几何：仅在有效切片时挂载，与原场景对象完全隔离 */}
        {sliceData && <SliceGeometry data={sliceData} opacity={sliceOpacity} />}
      </group>

      {/* 现有参考晶面：透明度独立，不受切片影响 */}
      <mesh position={[0, 0, 0]}>
        <planeGeometry args={[5.6, 5.6]} />
        <meshBasicMaterial color="#69cde2" transparent opacity={planeOpacity} side={THREE.DoubleSide} />
      </mesh>

      <Grid
        args={[8, 8]}
        cellSize={1.25}
        sectionSize={2.5}
        cellColor="#326778"
        sectionColor="#72b9c9"
        position={[0, -2.55, 0]}
      />
      <OrbitControls ref={controlsRef} makeDefault minDistance={5} maxDistance={15} />
      <SliceCameraRig
        controlsRef={controlsRef}
        enabled={sliceEnabled}
        normalKey={sliceNormalKey}
        flyNonce={flyNonce}
        worldNormal={worldNormal}
        worldTarget={worldTarget}
        valid={sliceData !== null}
      />
    </Canvas>
  );
}

function DefectCard({
  type,
  active,
  onClick,
  title,
  subtitle,
  color,
}: {
  type: DefectType;
  active: boolean;
  onClick: () => void;
  title: string;
  subtitle: string;
  color: string;
}) {
  return (
    <button className={active ? 'defect-card active' : 'defect-card'} onClick={onClick}>
      <span className="defect-symbol" style={{ color }}>
        {type === 'vacancy' ? '○' : type === 'substitution' ? '◉' : '∿'}
      </span>
      <span>
        <b>{title}</b>
        <small>{subtitle}</small>
      </span>
      <i>{active ? 'ACTIVE' : 'SELECT'}</i>
    </button>
  );
}

const SLICE_MODE_OPTIONS: { id: SliceMode; label: string }[] = [
  { id: 'x', label: 'X' },
  { id: 'y', label: 'Y' },
  { id: 'z', label: 'Z' },
  { id: 'custom', label: '自定义' },
];

export default function App() {
  const [cells, setCells] = useState(4);
  const [defect, setDefect] = useState<DefectType>('vacancy');
  const [showBonds, setShowBonds] = useState(true);
  const [planeOpacity, setPlaneOpacity] = useState(0.08);
  const [notice, setNotice] = useState('');

  const [sliceEnabled, setSliceEnabled] = useState(false);
  const [sliceMode, setSliceMode] = useState<SliceMode>('z');
  const [sliceOffset, setSliceOffset] = useState(0);
  const [sliceOpacity, setSliceOpacity] = useState(0.16);
  const [customNormal, setCustomNormal] = useState<Record<AxisId, number>>({ x: 1, y: 1, z: 0 });
  const [flyNonce, setFlyNonce] = useState(0);

  const metrics = useMemo(
    () => ({
      atoms: defect === 'vacancy' ? cells ** 3 - 1 : cells ** 3,
      neighbors: defect === 'dislocation' || defect === 'vacancy' ? '5 / 6' : '6 / 6',
      strain: defect === 'dislocation' ? '8.4%' : defect === 'substitution' ? '3.1%' : '1.7%',
    }),
    [cells, defect],
  );

  const extent = useMemo(() => structureHalfExtent(cells), [cells]);
  const sliderMax = useMemo(() => Math.ceil((extent + 0.5) * 2) / 2, [extent]);

  const localNormal = useMemo(() => resolveNormal(sliceMode, customNormal), [sliceMode, customNormal]);
  const sliceData = useMemo(
    () => (sliceEnabled && localNormal ? computeSlice(cells, defect, localNormal, sliceOffset) : null),
    [sliceEnabled, localNormal, cells, defect, sliceOffset],
  );
  const sliceEmpty = sliceEnabled && sliceData === null;
  const normalInvalid = sliceEnabled && localNormal === null;

  // 相机参数变换到世界空间（晶格组带固定旋转）
  const worldNormal = useMemo(
    () => (localNormal ? localNormal.clone().applyQuaternion(LATTICE_QUATERNION) : null),
    [localNormal],
  );
  const worldTarget = useMemo(
    () => (sliceData ? sliceData.anchor.clone().applyQuaternion(LATTICE_QUATERNION) : null),
    [sliceData],
  );
  const sliceNormalKey =
    sliceMode === 'custom'
      ? `custom-${customNormal.x},${customNormal.y},${customNormal.z}`
      : sliceMode;

  const selectSliceMode = (mode: SliceMode) => {
    setSliceMode(mode);
    setSliceOffset(0);
  };

  const planeEquation =
    sliceMode === 'custom' && localNormal
      ? `n·r = ${sliceOffset.toFixed(2)}`
      : `${sliceMode.toUpperCase()} = ${sliceOffset.toFixed(2)}`;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-icon">◈</div>
          <div>
            <span className="eyebrow">MATERIALS SCIENCE / LATTICE LAB</span>
            <h1>
              Crystal Defect Lab <span>·</span> 晶格缺陷分析
            </h1>
          </div>
        </div>
        <div className="header-meta">
          <span className="sample">SAMPLE / FCC-AL-04</span>
          <button onClick={() => setNotice('结构快照已准备，可接入数据导出')}>导出快照 ↗</button>
        </div>
      </header>

      <main className="layout">
        <section className="model-card">
          <div className="model-head">
            <div>
              <span className="eyebrow">UNIT CELL EXPLORER</span>
              <h2>面心立方晶格 · 缺陷结构可视化</h2>
            </div>
            <div className="head-stats">
              <span>
                <b>{cells}³</b> 晶胞
              </span>
              <span>
                <b>{metrics.atoms}</b> 原子
              </span>
            </div>
          </div>
          <div className="canvas-wrap">
            <DefectScene
              cells={cells}
              defect={defect}
              showBonds={showBonds}
              planeOpacity={planeOpacity}
              sliceData={sliceData}
              sliceOpacity={sliceOpacity}
              sliceEnabled={sliceEnabled}
              sliceNormalKey={sliceNormalKey}
              flyNonce={flyNonce}
              worldNormal={worldNormal}
              worldTarget={worldTarget}
            />
            {sliceEmpty && (
              <div className="slice-empty">
                <b>切片为空</b>
                <span>
                  {normalInvalid
                    ? '自定义法向量为零向量，请输入有效方向。'
                    : `切片未穿过结构（结构范围 ±${extent.toFixed(2)}，当前 d = ${sliceOffset.toFixed(2)}）。`}
                </span>
                <small>原场景未改动，调整偏移量或法向后即可恢复截面。</small>
              </div>
            )}
            <div className="model-legend">
              <span>
                <i className="dot host" />
                基体原子
              </span>
              <span>
                <i className="dot sub" />
                缺陷原子
              </span>
              <span>
                <i className="dot line" />
                滑移/缺陷线
              </span>
              {sliceEnabled && sliceData && (
                <span>
                  <i className="dot slice" />
                  切片截面
                </span>
              )}
            </div>
            <div className="model-hint">拖拽旋转 · 滚轮缩放 · 三维晶面已叠加</div>
          </div>
        </section>

        <aside className="sidebar">
          <section className="panel summary">
            <span className="eyebrow">STRUCTURE SUMMARY</span>
            <h3>晶格参数</h3>
            <div className="summary-grid">
              <div>
                <span>晶格类型</span>
                <b>FCC</b>
              </div>
              <div>
                <span>晶格常数</span>
                <b>4.05 Å</b>
              </div>
              <div>
                <span>理论密度</span>
                <b>2.70 g/cm³</b>
              </div>
              <div>
                <span>当前温度</span>
                <b>298 K</b>
              </div>
            </div>
          </section>

          <section className="panel">
            <div className="panel-title">
              <div>
                <span className="eyebrow">DEFECT LIBRARY</span>
                <h3>缺陷类型</h3>
              </div>
              <span className="count">3 MODELS</span>
            </div>
            <div className="defect-list">
              <DefectCard
                type="vacancy"
                active={defect === 'vacancy'}
                onClick={() => setDefect('vacancy')}
                title="空位缺陷"
                subtitle="原子位置缺失"
                color={colors.vacancy}
              />
              <DefectCard
                type="substitution"
                active={defect === 'substitution'}
                onClick={() => setDefect('substitution')}
                title="替位缺陷"
                subtitle="异质原子替换"
                color={colors.substitution}
              />
              <DefectCard
                type="dislocation"
                active={defect === 'dislocation'}
                onClick={() => setDefect('dislocation')}
                title="刃型位错"
                subtitle="局部滑移面错位"
                color="#ef8d86"
              />
            </div>
          </section>

          <section className="panel controls">
            <div className="panel-title">
              <div>
                <span className="eyebrow">ANALYSIS CONTROLS</span>
                <h3>分析参数</h3>
              </div>
            </div>
            <label>
              晶胞数量 <output>{cells} × {cells} × {cells}</output>
              <input
                type="range"
                min="3"
                max="5"
                value={cells}
                onChange={event => setCells(Number(event.target.value))}
              />
            </label>
            <label>
              晶面透明度 <output>{Math.round(planeOpacity * 100)}%</output>
              <input
                type="range"
                min="0"
                max="0.3"
                step="0.01"
                value={planeOpacity}
                onChange={event => setPlaneOpacity(Number(event.target.value))}
              />
            </label>
            <button className={showBonds ? 'toggle active' : 'toggle'} onClick={() => setShowBonds(value => !value)}>
              <span className="toggle-dot" />
              显示最近邻连接
            </button>
          </section>

          <section className="panel slice-panel">
            <div className="panel-title">
              <div>
                <span className="eyebrow">PLANE SLICING</span>
                <h3>晶面切片观察</h3>
              </div>
              <span className={sliceEnabled ? 'count active' : 'count'}>{sliceEnabled ? 'SLICE ON' : 'SLICE OFF'}</span>
            </div>
            <button
              className={sliceEnabled ? 'toggle active' : 'toggle'}
              onClick={() => setSliceEnabled(value => !value)}
            >
              <span className="toggle-dot" />
              沿固定晶面定位相机观察缺陷
            </button>

            <div className={sliceEnabled ? 'slice-body' : 'slice-body disabled'}>
              <div className="seg-label">晶面法向</div>
              <div className="segmented">
                {SLICE_MODE_OPTIONS.map(option => (
                  <button
                    key={option.id}
                    className={sliceMode === option.id ? 'seg active' : 'seg'}
                    onClick={() => selectSliceMode(option.id)}
                    tabIndex={sliceEnabled ? 0 : -1}
                  >
                    {option.label}
                  </button>
                ))}
              </div>

              {sliceMode === 'custom' && (
                <div className="custom-normal">
                  {(['x', 'y', 'z'] as const).map(axis => (
                    <label key={axis}>
                      {axis}
                      <input
                        type="number"
                        step="0.5"
                        value={customNormal[axis]}
                        onChange={event =>
                          setCustomNormal(prev => ({ ...prev, [axis]: Number(event.target.value) || 0 }))
                        }
                      />
                    </label>
                  ))}
                  {normalInvalid && <small className="field-warn">法向量不能为零向量</small>}
                </div>
              )}

              <label>
                平面偏移 d <output>{sliceOffset.toFixed(2)}</output>
                <input
                  type="range"
                  min={-sliderMax}
                  max={sliderMax}
                  step="0.05"
                  value={sliceOffset}
                  onChange={event => setSliceOffset(Number(event.target.value))}
                />
                <small className="range-ext">结构范围 ±{extent.toFixed(2)} · 超出显示空态</small>
              </label>

              <label>
                截面不透明度 <output>{Math.round(sliceOpacity * 100)}%</output>
                <input
                  type="range"
                  min="0"
                  max="0.6"
                  step="0.01"
                  value={sliceOpacity}
                  onChange={event => setSliceOpacity(Number(event.target.value))}
                />
              </label>

              <div className={sliceEmpty ? 'slice-status warn' : 'slice-status'}>
                {sliceEmpty ? (
                  <span>空态：{normalInvalid ? '法向量无效' : '切片未穿过结构'}</span>
                ) : (
                  <>
                    <span>{planeEquation}</span>
                    <span>
                      截到原子 <b>{sliceData?.atomCuts.length ?? 0}</b> · 键{' '}
                      <b>{(sliceData?.bondCuts.length ?? 0) + (sliceData?.bondSegments.length ?? 0)}</b>
                    </span>
                  </>
                )}
              </div>

              <button
                className="ghost-btn"
                disabled={sliceEmpty}
                onClick={() => setFlyNonce(value => value + 1)}
              >
                重新沿法向定位相机 ↥
              </button>
            </div>
          </section>

          <section className="panel metrics-panel">
            <span className="eyebrow">DEFECT METRICS</span>
            <h3>结构指标</h3>
            <div className="metric-row">
              <div>
                <span>原子数</span>
                <b>{metrics.atoms}</b>
              </div>
              <div>
                <span>平均配位</span>
                <b>{metrics.neighbors}</b>
              </div>
              <div>
                <span>局部应变</span>
                <b>{metrics.strain}</b>
              </div>
            </div>
            <p>指标为演示计算结果，后续可接入真实晶体结构文件与模拟数据。</p>
          </section>

          {notice && (
            <div className="toast" onClick={() => setNotice('')}>
              {notice}
              <span>×</span>
            </div>
          )}
        </aside>
      </main>
    </div>
  );
}
