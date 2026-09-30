import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { Grid, Html, Line, OrbitControls } from '@react-three/drei';
import * as THREE from 'three';

type DefectType = 'vacancy' | 'substitution' | 'dislocation';
type PlaneDirection = 'x' | 'y' | 'z' | 'custom';
type AtomPoint = {
  key: string;
  x: number;
  y: number;
  z: number;
  defect?: 'vacancy' | 'substitution' | 'normal';
};

type SliceGeometry = {
  kind: 'geometry';
  center: THREE.Vector3;
  localNormal: THREE.Vector3;
  worldCenter: THREE.Vector3;
  worldNormal: THREE.Vector3;
  points: THREE.Vector3[];
  offset: number;
  label: string;
};

type SliceStatus = {
  kind: 'empty' | 'invalid';
  message: string;
};

type SliceResult = SliceGeometry | SliceStatus | null;
type CameraTarget = { center: THREE.Vector3; normal: THREE.Vector3 };

const colors = {
  host: '#64c9e1',
  substitution: '#efb863',
  vacancy: '#ef7e83',
  bond: '#4b8190',
  slice: '#8df3ff',
};

const SPACING = 1.25;
const STRUCTURE_QUATERNION = new THREE.Quaternion().setFromEuler(new THREE.Euler(0.15, -0.35, 0));
const DIRECTION_NORMALS: Record<Exclude<PlaneDirection, 'custom'>, THREE.Vector3> = {
  x: new THREE.Vector3(1, 0, 0),
  y: new THREE.Vector3(0, 1, 0),
  z: new THREE.Vector3(0, 0, 1),
};
const DIRECTION_LABELS: Record<Exclude<PlaneDirection, 'custom'>, string> = {
  x: '(100)',
  y: '(010)',
  z: '(001)',
};

function getNormal(direction: PlaneDirection, custom: [string, string, string]): THREE.Vector3 | null {
  const normal =
    direction === 'custom'
      ? new THREE.Vector3(Number.parseFloat(custom[0]), Number.parseFloat(custom[1]), Number.parseFloat(custom[2]))
      : DIRECTION_NORMALS[direction].clone();
  const length = normal.length();
  if (!Number.isFinite(length) || length < 1e-6) return null;
  return normal.divideScalar(length);
}

function formatVector(vector: THREE.Vector3) {
  return `${vector.x.toFixed(2)}, ${vector.y.toFixed(2)}, ${vector.z.toFixed(2)}`;
}

function structureHalfSize(cells: number) {
  return ((cells - 1) * SPACING) / 2;
}

function defectLocalPoint(cells: number, defect: DefectType) {
  const mid = (cells - 1) / 2;
  const latticePoint = defect === 'dislocation' ? { x: 2, y: 0, z: 2 } : { x: 1, y: 1, z: 1 };
  return new THREE.Vector3(
    (latticePoint.x - mid) * SPACING,
    (latticePoint.y - mid) * SPACING,
    (latticePoint.z - mid) * SPACING,
  );
}

function calculateSlice(
  direction: PlaneDirection,
  custom: [string, string, string],
  offset: number,
  cells: number,
): SliceResult {
  const normal = getNormal(direction, custom);
  if (!normal) return { kind: 'invalid', message: '自定义法向量不能为零向量，切片未应用' };

  const label =
    direction === 'custom' ? `n = (${formatVector(normal)})` : `法向 ${direction.toUpperCase()} / ${DIRECTION_LABELS[direction]}`;
  const half = structureHalfSize(cells);
  const min = new THREE.Vector3(-half, -half, -half);
  const max = new THREE.Vector3(half, half, half);
  const support = half * (Math.abs(normal.x) + Math.abs(normal.y) + Math.abs(normal.z));

  if (offset < -support - 1e-5 || offset > support + 1e-5) {
    return {
      kind: 'empty',
      message: `空态：切片超出结构范围（d=${offset.toFixed(2)}，有效范围约 ±${support.toFixed(2)}），原场景未修改`,
    };
  }

  const intersections: THREE.Vector3[] = [];
  const corner = (x: 0 | 1, y: 0 | 1, z: 0 | 1) =>
    new THREE.Vector3(x ? max.x : min.x, y ? max.y : min.y, z ? max.z : min.z);
  const addEdgeIntersection = (a: THREE.Vector3, b: THREE.Vector3) => {
    const da = normal.dot(a) - offset;
    const db = normal.dot(b) - offset;
    if (Math.abs(db - da) < 1e-6) {
      if (Math.abs(da) < 1e-6) {
        intersections.push(a.clone(), b.clone());
      }
      return;
    }
    if (da * db <= 1e-8) {
      const t = THREE.MathUtils.clamp(-da / (db - da), 0, 1);
      intersections.push(a.clone().lerp(b, t));
    }
  };

  for (let y = 0 as 0 | 1; y <= 1; y += 1) {
    for (let z = 0 as 0 | 1; z <= 1; z += 1) {
      addEdgeIntersection(corner(0, y, z), corner(1, y, z));
    }
  }
  for (let x = 0 as 0 | 1; x <= 1; x += 1) {
    for (let z = 0 as 0 | 1; z <= 1; z += 1) {
      addEdgeIntersection(corner(x, 0, z), corner(x, 1, z));
    }
  }
  for (let x = 0 as 0 | 1; x <= 1; x += 1) {
    for (let y = 0 as 0 | 1; y <= 1; y += 1) {
      addEdgeIntersection(corner(x, y, 0), corner(x, y, 1));
    }
  }

  const points = intersections.filter(
    (point, index) => intersections.findIndex(other => other.distanceToSquared(point) < 1e-8) === index,
  );
  const center = normal.clone().multiplyScalar(offset);
  const reference = Math.abs(normal.y) > 0.9 ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 1, 0);
  const u = new THREE.Vector3().crossVectors(reference, normal).normalize();
  const v = new THREE.Vector3().crossVectors(normal, u).normalize();

  points.sort((a, b) => {
    const pa = a.clone().sub(center);
    const pb = b.clone().sub(center);
    return Math.atan2(pa.dot(v), pa.dot(u)) - Math.atan2(pb.dot(v), pb.dot(u));
  });

  let area = 0;
  for (let i = 0; i < points.length; i += 1) {
    const a = points[i].clone().sub(center);
    const b = points[(i + 1) % points.length].clone().sub(center);
    area += a.dot(u) * b.dot(v) - b.dot(u) * a.dot(v);
  }
  area = Math.abs(area) / 2;
  if (points.length < 3 || area < 1e-7) {
    return {
      kind: 'empty',
      message: `空态：切片仅接触结构边界（d=${offset.toFixed(2)}），未生成截面且原场景未修改`,
    };
  }

  return {
    kind: 'geometry',
    center,
    localNormal: normal,
    worldCenter: center.clone().applyQuaternion(STRUCTURE_QUATERNION),
    worldNormal: normal.clone().applyQuaternion(STRUCTURE_QUATERNION),
    points,
    offset,
    label,
  };
}

function buildAtoms(cells: number, defect: DefectType): AtomPoint[] {
  const atoms: AtomPoint[] = [];
  for (let x = 0; x < cells; x += 1) {
    for (let y = 0; y < cells; y += 1) {
      for (let z = 0; z < cells; z += 1) {
        const key = `${x}-${y}-${z}`;
        if (defect === 'vacancy' && x === 1 && y === 1 && z === 1) continue;
        if (defect === 'substitution' && x === 1 && y === 1 && z === 1) {
          atoms.push({ key, x, y, z, defect: 'substitution' });
        } else {
          atoms.push({ key, x, y, z, defect: 'normal' });
        }
      }
    }
  }
  if (defect === 'dislocation') return atoms.filter(atom => !(atom.x === 2 && atom.y === 0 && atom.z === 2));
  return atoms;
}

function atomPosition(atom: Pick<AtomPoint, 'x' | 'y' | 'z'>, cells: number) {
  const mid = (cells - 1) / 2;
  return new THREE.Vector3((atom.x - mid) * SPACING, (atom.y - mid) * SPACING, (atom.z - mid) * SPACING);
}

function CameraRig({ active, nonce, target }: { active: boolean; nonce: number; target: CameraTarget | null }) {
  const controlsRef = useRef<any>(null);
  const camera = useThree(state => state.camera);
  const targetRef = useRef(target);
  const savedViewRef = useRef<{ position: THREE.Vector3; target: THREE.Vector3; up: THREE.Vector3 } | null>(null);
  targetRef.current = target;

  useEffect(() => {
    const controls = controlsRef.current;
    if (!controls) return;
    let frame = 0;

    const animateTo = (
      endPosition: THREE.Vector3,
      endTarget: THREE.Vector3,
      endUp: THREE.Vector3,
      onComplete?: () => void,
    ) => {
      const start = performance.now();
      const duration = 560;
      const startPosition = camera.position.clone();
      const startTarget = controls.target.clone();
      const startUp = camera.up.clone();
      const ease = (t: number) => (t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2);

      const tick = (now: number) => {
        const raw = Math.min(1, (now - start) / duration);
        const t = ease(raw);
        camera.position.lerpVectors(startPosition, endPosition, t);
        controls.target.lerpVectors(startTarget, endTarget, t);
        camera.up.lerpVectors(startUp, endUp, t).normalize();
        camera.lookAt(controls.target);
        controls.update();
        if (raw < 1) frame = requestAnimationFrame(tick);
        else onComplete?.();
      };
      frame = requestAnimationFrame(tick);
    };

    if (!active) {
      const saved = savedViewRef.current;
      if (saved) {
        animateTo(saved.position, saved.target, saved.up, () => {
          savedViewRef.current = null;
        });
      }
      return () => cancelAnimationFrame(frame);
    }

    const sliceTarget = targetRef.current;
    if (!sliceTarget || nonce === 0) return () => cancelAnimationFrame(frame);
    if (!savedViewRef.current) {
      savedViewRef.current = {
        position: camera.position.clone(),
        target: controls.target.clone(),
        up: camera.up.clone(),
      };
    }

    const desiredTarget = sliceTarget.center.clone();
    const desiredPosition = desiredTarget.clone().addScaledVector(sliceTarget.normal, 10.5);
    const worldUp = new THREE.Vector3(0, 1, 0);
    const desiredUp =
      Math.abs(sliceTarget.normal.dot(worldUp)) > 0.92
        ? new THREE.Vector3(0, 0, 1)
        : worldUp.clone().sub(sliceTarget.normal.clone().multiplyScalar(sliceTarget.normal.dot(worldUp))).normalize();

    animateTo(desiredPosition, desiredTarget, desiredUp);
    return () => cancelAnimationFrame(frame);
  }, [active, nonce, camera]);

  return <OrbitControls ref={controlsRef} makeDefault minDistance={5} maxDistance={18} />;
}

function SliceSection({ slice, opacity }: { slice: SliceGeometry; opacity: number }) {
  const geometry = useMemo(() => {
    const buffer = new THREE.BufferGeometry();
    const positions = new Float32Array(slice.points.length * 3);
    slice.points.forEach((point, index) => {
      positions[index * 3] = point.x - slice.center.x;
      positions[index * 3 + 1] = point.y - slice.center.y;
      positions[index * 3 + 2] = point.z - slice.center.z;
    });
    buffer.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    const indices: number[] = [];
    for (let index = 1; index < slice.points.length - 1; index += 1) {
      indices.push(0, index, index + 1);
    }
    buffer.setIndex(indices);
    buffer.computeVertexNormals();
    return buffer;
  }, [slice]);

  useEffect(() => () => geometry.dispose(), [geometry]);

  return (
    <group>
      <mesh geometry={geometry} position={slice.center} renderOrder={10}>
        <meshBasicMaterial
          color={colors.slice}
          transparent
          opacity={opacity}
          side={THREE.DoubleSide}
          depthWrite={false}
        />
      </mesh>
      <Line
        points={[...slice.points, slice.points[0]]}
        color="#d6fbff"
        lineWidth={1.4}
        transparent
        opacity={Math.max(0.45, opacity + 0.28)}
      />
    </group>
  );
}

function NormalAnnotation({ slice }: { slice: SliceGeometry }) {
  const arrowStart = useMemo(
    () => slice.center.clone().addScaledVector(slice.localNormal, 1.05),
    [slice],
  );
  const arrow = useMemo(() => {
    const helper = new THREE.ArrowHelper(
      slice.localNormal,
      arrowStart,
      1.35,
      new THREE.Color('#ffe082'),
      0.28,
      0.13,
    );
    helper.renderOrder = 30;
    return helper;
  }, [arrowStart, slice.localNormal]);

  useEffect(() => {
    const lineMaterial = arrow.line.material as THREE.LineBasicMaterial;
    const coneMaterial = arrow.cone.material as THREE.MeshBasicMaterial;
    lineMaterial.depthTest = false;
    coneMaterial.depthTest = false;
    arrow.line.renderOrder = 30;
    arrow.cone.renderOrder = 30;
    return () => arrow.dispose();
  }, [arrow]);

  const labelPosition = arrowStart.clone().addScaledVector(slice.localNormal, 1.72);

  return (
    <group>
      <primitive object={arrow} />
      <Html position={labelPosition.toArray()} center zIndexRange={[30, 0]} style={{ pointerEvents: 'none' }}>
        <span className="slice-normal-label">{slice.label}</span>
      </Html>
    </group>
  );
}

function DefectScene({
  cells,
  defect,
  showBonds,
  planeOpacity,
  slice,
  sliceOpacity,
  cameraNonce,
}: {
  cells: number;
  defect: DefectType;
  showBonds: boolean;
  planeOpacity: number;
  slice: SliceResult;
  sliceOpacity: number;
  cameraNonce: number;
}) {
  const clipGroupRef = useRef<THREE.Group>(null);
  const { gl } = useThree();
  const atoms = useMemo(() => buildAtoms(cells, defect), [cells, defect]);
  const bonds = useMemo(() => {
    const list: [THREE.Vector3, THREE.Vector3][] = [];
    atoms.forEach(atom => {
      const origin = atomPosition(atom, cells);
      const xNeighbor = atoms.find(item => item.x === atom.x + 1 && item.y === atom.y && item.z === atom.z);
      if (xNeighbor) list.push([origin, atomPosition(xNeighbor, cells)]);
      const zNeighbor = atoms.find(item => item.x === atom.x && item.y === atom.y && item.z === atom.z + 1);
      if (zNeighbor) list.push([origin, atomPosition(zNeighbor, cells)]);
    });
    return list;
  }, [atoms, cells]);

  const activeSlice = slice?.kind === 'geometry' ? slice : null;

  useLayoutEffect(() => {
    const group = clipGroupRef.current;
    if (!group || !activeSlice) return;
    const plane = new THREE.Plane(activeSlice.worldNormal, -activeSlice.offset);
    const previousClipping = gl.localClippingEnabled;
    const clippedMaterials: THREE.Material[] = [];

    gl.localClippingEnabled = true;
    group.traverse(object => {
      const material = (object as THREE.Mesh).material as THREE.Material | THREE.Material[] | undefined;
      if (!material) return;
      const materials = Array.isArray(material) ? material : [material];
      materials.forEach(item => {
        item.clippingPlanes = [plane];
        item.clipShadows = true;
        clippedMaterials.push(item);
      });
    });

    return () => {
      gl.localClippingEnabled = previousClipping;
      clippedMaterials.forEach(item => {
        item.clippingPlanes = [];
        item.clipShadows = false;
      });
    };
  }, [gl, activeSlice]);

  return (
    <Canvas shadows camera={{ position: [8, 6, 8], fov: 38 }}>
      <color attach="background" args={['#09151d']} />
      <ambientLight intensity={1.25} />
      <directionalLight position={[5, 8, 5]} intensity={2.3} castShadow />
      <pointLight position={[-4, 2, 3]} color="#5fd4ee" intensity={12} distance={12} />

      <group ref={clipGroupRef} quaternion={STRUCTURE_QUATERNION}>
        {showBonds &&
          bonds.map((bond, index) => (
            <Line key={index} points={bond} color={colors.bond} lineWidth={1.2} transparent opacity={0.65} />
          ))}
        {atoms.map(atom => {
          const position = atomPosition(atom, cells);
          const color = atom.defect === 'substitution' ? colors.substitution : colors.host;
          return (
            <mesh key={atom.key} position={position} castShadow>
              <sphereGeometry args={[atom.defect === 'substitution' ? 0.3 : 0.2, 18, 12]} />
              <meshStandardMaterial
                color={color}
                emissive={atom.defect === 'substitution' ? colors.substitution : '#000000'}
                emissiveIntensity={atom.defect === 'substitution' ? 0.35 : 0}
              />
            </mesh>
          );
        })}
        {defect === 'vacancy' && (
          <mesh position={[0, 0, 0]}>
            <sphereGeometry args={[0.35, 16, 12]} />
            <meshBasicMaterial color={colors.vacancy} transparent opacity={0.26} wireframe />
          </mesh>
        )}
        {defect === 'dislocation' && (
          <Line
            points={[
              new THREE.Vector3(-1.9, -1.9, 0),
              new THREE.Vector3(-0.65, -1.9, 0),
              new THREE.Vector3(-0.65, -0.65, 0),
              new THREE.Vector3(0.65, -0.65, 0),
              new THREE.Vector3(0.65, 0.65, 0),
              new THREE.Vector3(1.9, 0.65, 0),
            ]}
            color="#f08d87"
            lineWidth={4}
          />
        )}
      </group>

      {activeSlice && (
        <group quaternion={STRUCTURE_QUATERNION}>
          <SliceSection slice={activeSlice} opacity={sliceOpacity} />
          <NormalAnnotation slice={activeSlice} />
        </group>
      )}

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
      <CameraRig
        active={slice?.kind === 'geometry'}
        nonce={cameraNonce}
        target={activeSlice ? { center: activeSlice.worldCenter, normal: activeSlice.worldNormal } : null}
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

export default function App() {
  const [cells, setCells] = useState(4);
  const [defect, setDefect] = useState<DefectType>('vacancy');
  const [showBonds, setShowBonds] = useState(true);
  const [planeOpacity, setPlaneOpacity] = useState(0.08);
  const [notice, setNotice] = useState('');

  const [sliceEnabled, setSliceEnabled] = useState(false);
  const [planeDirection, setPlaneDirection] = useState<PlaneDirection>('z');
  const [customNormal, setCustomNormal] = useState<[string, string, string]>(['1', '1', '1']);
  const [sliceOffset, setSliceOffset] = useState(0);
  const [sliceOpacity, setSliceOpacity] = useState(0.18);
  const [autoCamera, setAutoCamera] = useState(true);
  const [cameraNonce, setCameraNonce] = useState(0);

  const metrics = useMemo(
    () => ({
      atoms: defect === 'vacancy' ? cells ** 3 - 1 : cells ** 3,
      neighbors: defect === 'dislocation' || defect === 'vacancy' ? '5 / 6' : '6 / 6',
      strain: defect === 'dislocation' ? '8.4%' : defect === 'substitution' ? '3.1%' : '1.7%',
    }),
    [cells, defect],
  );
  const currentNormal = useMemo(() => getNormal(planeDirection, customNormal), [planeDirection, customNormal]);
  const sliceResult = useMemo(
    () => (sliceEnabled ? calculateSlice(planeDirection, customNormal, sliceOffset, cells) : null),
    [sliceEnabled, planeDirection, customNormal, sliceOffset, cells],
  );
  const defectPoint = useMemo(() => defectLocalPoint(cells, defect), [cells, defect]);

  const alignCamera = () => {
    if (sliceResult?.kind === 'geometry') setCameraNonce(value => value + 1);
  };
  const movePlaneToDefect = () => {
    if (!currentNormal) return;
    setSliceOffset(Number(currentNormal.dot(defectPoint).toFixed(3)));
    if (autoCamera) setCameraNonce(value => value + 1);
  };
  const changeDirection = (direction: PlaneDirection) => {
    setPlaneDirection(direction);
    if (!sliceEnabled) return;
    const normal = getNormal(direction, customNormal);
    if (normal) setSliceOffset(Number(normal.dot(defectPoint).toFixed(3)));
    if (autoCamera && normal) setCameraNonce(value => value + 1);
  };
  const toggleSlice = () => {
    const next = !sliceEnabled;
    setSliceEnabled(next);
    if (next && currentNormal) {
      setSliceOffset(Number(currentNormal.dot(defectPoint).toFixed(3)));
      if (autoCamera) setCameraNonce(value => value + 1);
    }
  };

  const sliceStatus =
    !sliceEnabled || !sliceResult
      ? '切片观察关闭 · 启用后沿当前缺陷所在晶面定位'
      : sliceResult.kind === 'geometry'
        ? `裁剪可见对象 · ${sliceResult.label} · n·r = ${sliceResult.offset.toFixed(2)}`
        : sliceResult.message;

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
              slice={sliceResult}
              sliceOpacity={sliceOpacity}
              cameraNonce={cameraNonce}
            />
            <div className={sliceEnabled && sliceResult?.kind !== 'geometry' ? 'slice-status warning' : 'slice-status'}>
              {sliceStatus}
            </div>
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
              <span>
                <i className="dot slice" />
                截面切片
              </span>
            </div>
            <div className="model-hint">拖拽旋转 · 滚轮缩放 · 固定晶面切片与法向标注</div>
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
              现有晶面透明度 <output>{Math.round(planeOpacity * 100)}%</output>
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

            <div className="slice-panel">
              <div className="subpanel-head">
                <div>
                  <span className="eyebrow">SLICE PLANE / FIXED VIEW</span>
                  <h4>固定晶面观察</h4>
                </div>
                <button className={sliceEnabled ? 'mini-toggle active' : 'mini-toggle'} onClick={toggleSlice}>
                  <span />
                  {sliceEnabled ? '已启用' : '启用'}
                </button>
              </div>

              <div className="mode-options">
                <button className="mode-option active" type="button">
                  <b>裁剪可见对象</b>
                  <i>已选用</i>
                </button>
                <button className="mode-option" type="button" disabled title="本版本固定使用裁剪模式">
                  <b>生成真实截面几何</b>
                  <i>未启用</i>
                </button>
              </div>

              <div className="direction-grid">
                {(['x', 'y', 'z', 'custom'] as PlaneDirection[]).map(direction => (
                  <button
                    key={direction}
                    className={planeDirection === direction ? 'direction-button active' : 'direction-button'}
                    onClick={() => changeDirection(direction)}
                  >
                    {direction === 'custom' ? '自定义 n' : `${direction.toUpperCase()} 法向`}
                    <small>{direction === 'custom' ? 'hkl' : DIRECTION_LABELS[direction as Exclude<PlaneDirection, 'custom'>]}</small>
                  </button>
                ))}
              </div>

              {planeDirection === 'custom' && (
                <div className="vector-inputs">
                  {(['X', 'Y', 'Z'] as const).map((axis, index) => (
                    <label key={axis}>
                      <span>{axis}</span>
                      <input
                        type="number"
                        step="0.1"
                        value={customNormal[index]}
                        onChange={event =>
                          setCustomNormal(current => {
                            const next: [string, string, string] = [...current] as [string, string, string];
                            next[index] = event.target.value;
                            return next;
                          })
                        }
                      />
                    </label>
                  ))}
                </div>
              )}

              <label>
                切片偏移 d <output>{sliceOffset.toFixed(2)}</output>
                <input
                  type="range"
                  min="-8"
                  max="8"
                  step="0.05"
                  value={sliceOffset}
                  onChange={event => setSliceOffset(Number(event.target.value))}
                />
              </label>
              <div className="slice-actions">
                <button className="mini-action" type="button" disabled={!sliceEnabled || !currentNormal} onClick={movePlaneToDefect}>
                  对准当前缺陷
                </button>
                <button
                  className="mini-action"
                  type="button"
                  disabled={!sliceEnabled || sliceResult?.kind !== 'geometry'}
                  onClick={alignCamera}
                >
                  定位相机
                </button>
              </div>
              <label>
                切片透明度 <output>{Math.round(sliceOpacity * 100)}%</output>
                <input
                  type="range"
                  min="0"
                  max="0.6"
                  step="0.01"
                  value={sliceOpacity}
                  onChange={event => setSliceOpacity(Number(event.target.value))}
                />
              </label>
              <button
                className={autoCamera ? 'toggle active compact' : 'toggle compact'}
                onClick={() => {
                  const next = !autoCamera;
                  setAutoCamera(next);
                  if (next && sliceEnabled) setCameraNonce(value => value + 1);
                }}
              >
                <span className="toggle-dot" />
                切换晶面时自动定位相机
              </button>
              <p className="slice-note">
                切片使用独立透明度，不影响原有参考晶面；超界仅显示空态。关闭切片后相机恢复到启用前视图。
              </p>
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
