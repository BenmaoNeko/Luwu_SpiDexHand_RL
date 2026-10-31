# Third-party software

- **cube.js 1.3.2** — https://github.com/ldez/cubejs — MIT license. The dependency supplies the cube model and Kociemba two-phase solver. The Vite compatibility wrapper only provides an explicit legacy UMD context; the solving algorithm is unchanged.
- **Three.js 0.186.1** — https://github.com/mrdoob/three.js — MIT license. Supplies rendering, geometry, OrbitControls, vectors, matrices and quaternions.
- **Vite 8.3.1** — https://github.com/vitejs/vite — MIT license. Development and build tooling.

Exact versions, including transitive dependencies, are locked in `package-lock.json`. License texts shipped by runtime dependencies are copied to `licenses/` and retained with this project. Development dependency license texts remain in their installed npm packages. No network-hosted scripts are loaded by the built application.
