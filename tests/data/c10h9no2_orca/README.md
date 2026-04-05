This folder contains a heavier ORCA fixture candidate derived from the user-provided Chem3D V3000 MOL file.

Source geometry
- Original file: `C:\Users\user\Downloads\Chem3D XML в.mol`
- Parsed formula: `C10H9NO2`
- Assumed charge/multiplicity: `0 1`

Files
- `c10h9no2_initial.xyz`: Cartesian geometry extracted from the MOL file.
- `00_opt_r2scan3c.inp`: cheap and robust geometry optimization.
- `10_sp_props.inp`: single-point run for charges, bond orders, Molden export, and density / HOMO / LUMO cubes.
- `20_fukui_neutral.inp`: neutral single-point at the optimized geometry.
- `21_fukui_anion.inp`: anion single-point at the optimized geometry.
- `22_fukui_cation.inp`: cation single-point at the optimized geometry.

Suggested workflow
1. Run `00_opt_r2scan3c.inp`.
2. Save the final optimized structure as `c10h9no2_opt.xyz`.
3. Run `10_sp_props.inp`.
4. Run `orca_2mkl 10_sp_props -molden` to obtain a Molden file.
5. Run the three Fukui jobs and derive condensed Fukui indices from atomic charges:
   - `f_plus(i)  = q_i(N) - q_i(N+1)`
   - `f_minus(i) = q_i(N-1) - q_i(N)`
   - `dual_descriptor(i) = f_plus(i) - f_minus(i)`

What ORCA should give us directly
- ORCA output: `.out`
- Wavefunction / density container: `.gbw`, `.scfp`
- XYZ coordinates: with `XYZFile`
- Molden: via `orca_2mkl ... -molden`
- Cube files for electron density and selected MOs: via `%plots`

Important limitation
- This machine has ORCA 5.0.4. It is good enough for ORCA `.out`, `.xyz`, `.molden`, density cubes, and MO cubes.
- ESP cube generation via `orca_plot` is documented as available starting with ORCA 6.1, so the ESP fixture will likely need either a newer ORCA or a separate post-processing step.
