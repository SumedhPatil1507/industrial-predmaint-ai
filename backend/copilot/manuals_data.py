"""
Comprehensive industrial equipment manuals and standard operating procedures (SOPs).
Indexed into vector store per machine type for RAG retrieval in Maintenance Copilot.
"""

EQUIPMENT_MANUALS = [
    # ── CNC Lathe ─────────────────────────────────────────────────────────────
    {
        "manual_id": "CNC-SOP-001",
        "machine_type": "CNC Lathe",
        "subsystem": "Spindle & Bearings",
        "title": "Main Spindle Bearing Overheating and High Vibration Remediation",
        "failure_modes": [
            "Bearing overheating (>80°C)",
            "High horizontal vibration (>6.0 mm/s)",
            "Spindle cage fatigue",
            "Acoustic hum and thermal expansion"
        ],
        "symptoms": "Spindle temperature exceeds 80°C with elevated horizontal vibration (vibration_h_mms > 6.0) and high power draw.",
        "root_cause_analysis": "Typically caused by lubrication starvation, contaminated grease (particulate ingress), or improper preload leading to angular contact bearing raceway spalling.",
        "sop_steps": [
            "1. Initiate Lockout/Tagout (LOTO) on main 415V 3-phase power isolator.",
            "2. Allow spindle housing to cool below 40°C before opening the headstock inspection port.",
            "3. Inspect spindle lube oil / grease color and particulate contamination with an optical inspection kit.",
            "4. Check spindle radial and axial runout using a 0.001mm dial indicator (nominal axial float < 0.005mm).",
            "5. If grease is degraded, flush old lubricant using solvent ISO-VG-32 and repack with Klüber Isoflex NBU 15 (15g per bearing).",
            "6. Torque spindle retaining locknut to 120 Nm using torque wrench TW-400.",
            "7. Perform a 30-minute tiered run-in sequence (500 RPM -> 1500 RPM -> 3000 RPM) while monitoring thermal equilibrium."
        ],
        "safety_protocol": "Class 3 LOTO required. Wear heat-resistant nitrile gloves (EN 388) and safety glasses. Ensure zero mechanical stored energy in belt drive.",
        "required_parts": ["Spindle Bearing Matched Pair (SKF 7014 CD/P4A)", "Klüber Isoflex NBU 15 Grease", "Spindle Labyrinth Seal (NBR-70)"],
        "estimated_mttr_hours": 3.5,
        "clearance_tolerances": "Axial play: 0.003 - 0.006 mm; Radial runout: < 0.002 mm; Max allowable temp: 75°C."
    },
    {
        "manual_id": "CNC-SOP-002",
        "machine_type": "CNC Lathe",
        "subsystem": "Drive Motor & Belt",
        "title": "Drive Motor Thermal Overload & Shaft Misalignment Troubleshooting",
        "failure_modes": [
            "Motor casing temperature spike (>90°C)",
            "High vertical vibration (>5.0 mm/s)",
            "Power surge at steady state",
            "V-belt slippage or glazing"
        ],
        "symptoms": "Motor temperature exceeds 90°C, vibration_v_mms > 5.0, accompanied by elevated power consumption under moderate load.",
        "root_cause_analysis": "Motor-to-headstock belt overtensioning or angular shaft misalignment causing excessive radial load on motor drive-end bearings.",
        "sop_steps": [
            "1. De-energize machine and lock main breaker in OFF position.",
            "2. Remove belt guard and measure belt tension with sonic belt tension meter (target: 350 - 400 Hz).",
            "3. Perform laser alignment on motor and spindle pulleys (angular offset max 0.25 deg, parallel offset < 0.5 mm).",
            "4. Inspect motor cooling fan cowl and fins for metal chip accumulation and clean with compressed dry air (< 2 bar).",
            "5. Check electrical motor insulation resistance (megger test > 50 Mega-ohms at 500V DC).",
            "6. Replace drive belts as a matched set (Gates Quad-Power 4 XPA-1250) if cracks or glazing are visible."
        ],
        "safety_protocol": "Verify zero electrical voltage before opening motor junction box. Ensure pinch hazard protection around belt pulley.",
        "required_parts": ["Gates Quad-Power XPA-1250 Belt Set (Qty 3)", "Motor Drive-End Bearing 6309-2Z/C3", "Cooling Fan Cowl Fasteners"],
        "estimated_mttr_hours": 2.0,
        "clearance_tolerances": "Belt tension: 380 Hz +/- 20 Hz; Pulley alignment: < 0.2 mm parallel offset."
    },
    {
        "manual_id": "CNC-SOP-003",
        "machine_type": "CNC Lathe",
        "subsystem": "Hydraulic Chuck & Clamping",
        "title": "Hydraulic Clamping Pressure Loss and Cylinder Seal Degeneration",
        "failure_modes": [
            "Oil pressure drop (< 3.0 bar)",
            "Chuck unclamping during cut",
            "Hydraulic pump cavitation noise",
            "Foaming in reservoir"
        ],
        "symptoms": "Hydraulic oil pressure drops below 3.0 bar or fluctuates rapidly, triggering clamping interlock alarms.",
        "root_cause_analysis": "Piston rotary union seal degradation, directional valve spool sticking, or clogged suction strainer in the power pack.",
        "sop_steps": [
            "1. Stop machine cycle and depressurize hydraulic accumulator circuit via manual dump valve.",
            "2. Inspect pressure gauge and proportional relief valve setpoint (nominal: 25 - 30 bar on manifold).",
            "3. Check rotary cylinder rear collector drain for excessive oil bypass (> 50 mL/min indicates seal failure).",
            "4. Clean 25-micron suction strainer and replace return line filter cartridge (Parker 937-399Q).",
            "5. If cylinder rotary seal is leaking, dismantle rear union and install fresh Viton double-lip seal kit.",
            "6. Bleed air from chuck cylinder bleed ports until continuous bubble-free oil emerges."
        ],
        "safety_protocol": "Hydraulic pressure must be zeroed before loosening any hydraulic fitting. Use oil catchment tray.",
        "required_parts": ["Rotary Union Seal Kit (Kit #RU-CNC-75)", "Return Filter Element Parker 10-micron", "Hydraulic Oil ISO VG 46 (20L)"],
        "estimated_mttr_hours": 2.5,
        "clearance_tolerances": "System pressure: 28 bar +/- 2 bar; Drain leakage: < 20 mL/min."
    },

    # ── Hydraulic Press ────────────────────────────────────────────────────────
    {
        "manual_id": "HYD-SOP-001",
        "machine_type": "Hydraulic Press",
        "subsystem": "Main Ram & Hydraulic Power Unit",
        "title": "Hydraulic Pressure Loss, Ram Drift, and Pump Cavitation Rectification",
        "failure_modes": [
            "Hydraulic line pressure collapse (< 2.5 bar on pilot / < 120 bar on main)",
            "Excessive pump vibration and acoustic rattling",
            "Ram creeping under tonnage load",
            "High oil temperature (> 65°C)"
        ],
        "symptoms": "Oil pressure falls significantly below nominal operating range, pump housing vibration exceeds 5 mm/s, high oil temperature.",
        "root_cause_analysis": "Axial piston pump swashplate wear, relief valve poppet erosion, or oil aeration caused by suction line air ingress.",
        "sop_steps": [
            "1. Engage mechanical ram safety lock / safety catch blocks before entering press bed.",
            "2. Turn off HPU main pump and tag out control panel.",
            "3. Inspect hydraulic fluid level and check for milky oil or aeration in sight glass.",
            "4. Check main pressure relief valve cartridge (Rexroth DBDS-20) for debris or spring fatigue.",
            "5. Measure pump case drain flow (normal < 4 L/min at 200 bar; > 10 L/min indicates severe piston/barrel wear).",
            "6. Inspect suction hose clamp connections for vacuum air leaks.",
            "7. Retorque manifold block tie-rods to 180 Nm in cross-pattern.",
            "8. Refill with Shell Tellus S2 MX 46 fluid through a 3-micron kidney-loop filtration cart."
        ],
        "safety_protocol": "MANDATORY: Insert mechanical ram safety struts (die safety blocks) rated for 200 tons before servicing.",
        "required_parts": ["Main Relief Cartridge Rexroth DBDS-20", "Piston Pump Rebuild Kit (A10VSO-71)", "Piston Seal Set (Polypack 250mm)", "Hydraulic Filter 5-micron"],
        "estimated_mttr_hours": 4.5,
        "clearance_tolerances": "Max ram drift: < 0.5 mm / 10 min; Operating oil temp: 40°C - 55°C; Max case drain: 3.5 L/min."
    },
    {
        "manual_id": "HYD-SOP-002",
        "machine_type": "Hydraulic Press",
        "subsystem": "Guide Bushings & Cylinder Seals",
        "title": "Cylinder Packing Seal Extrusion and Platen Skew Alignment",
        "failure_modes": [
            "Severe external oil leak down ram shaft",
            "Uneven tonnage distribution / platen tilting",
            "High horizontal vibration on downstroke",
            "Scoring on chrome cylinder rod"
        ],
        "symptoms": "High vibration along horizontal axis, jerky slide motion, visible hydraulic leakage at the main gland ring.",
        "root_cause_analysis": "Platen gib guide wear causing eccentric loading, resulting in unilateral chevron packing extrusion.",
        "sop_steps": [
            "1. Lower ram to resting position on lower bolster and depressurize system.",
            "2. Measure 4-corner gib clearance using feeler gauges (target: 0.08 - 0.12 mm per side).",
            "3. Adjust brass gib shim packs to re-center platen parallelism within 0.05 mm across 1500 mm span.",
            "4. Unscrew gland retaining ring and extract damaged chevron V-packing set.",
            "5. Polish minor longitudinal scratches on chrome ram using 800-grit emery cloth with oil lubrication.",
            "6. Install new polyurethane rod seal with PTFE anti-extrusion back-up rings and retorque gland bolts to 95 Nm."
        ],
        "safety_protocol": "Use safety harness if working on top manifold platform. Secure gland ring with rigging hoist before unbolting.",
        "required_parts": ["Hallite 780 Compact Seal Kit", "Bronze Gib Wear Strips (Set of 4)", "PTFE Guide Rings 220mm ID"],
        "estimated_mttr_hours": 4.0,
        "clearance_tolerances": "Gib clearance: 0.10 mm +/- 0.02 mm; Platen parallelism: < 0.05 mm TIR."
    },

    # ── Milling Machine ────────────────────────────────────────────────────────
    {
        "manual_id": "MIL-SOP-001",
        "machine_type": "Milling Machine",
        "subsystem": "Spindle Cartridge & Tool Clamping",
        "title": "Milling Spindle Bearing Chatter, Runout, and Tool Unclamp Error",
        "failure_modes": [
            "Chatter marks on machined workpiece",
            "High vibration in both horizontal and vertical axes (>5.5 mm/s)",
            "Bearing temperature rise (>75°C)",
            "Drawbar spring stack fatigue"
        ],
        "symptoms": "Vibration levels climb over 5.5 mm/s under standard feed rates, bearing temp climbs, surface finish degrades.",
        "root_cause_analysis": "Spindle ceramic hybrid bearings damaged by tool impact/crash, or Belleville disc spring stack cracked causing loose toolholder taper engagement.",
        "sop_steps": [
            "1. Perform E-Stop lock out and isolate 400V power.",
            "2. Measure drawbar retention force using a digital gauge (target: 18 kN for BT-40 taper, 25 kN for BT-50).",
            "3. If retention force < 12 kN, remove drawbar cylinder and replace Belleville spring stack in alternating series/parallel config.",
            "4. Inspect internal ISO-40 spindle taper for fretting corrosion and clean with diamond taper wiper.",
            "5. Measure spindle taper runout at gauge line (max allowable 0.003 mm).",
            "6. Verify spindle chiller coolant temperature (setpoint 20°C +/- 1°C, flow > 6 L/min).",
            "7. Perform dynamic balancing with accelerometer trimming if residual unbalance > 0.5 g-mm."
        ],
        "safety_protocol": "Ensure tool changer carousel is in locked home position. Wear cut-resistant gloves when handling tooling.",
        "required_parts": ["Belleville Spring Stack BT-40 (Pack of 120)", "Spindle Bearing Set NSK 7010CTYNDBLP4", "Spindle Coolant Chiller Filter"],
        "estimated_mttr_hours": 3.0,
        "clearance_tolerances": "Drawbar force: 18 kN +/- 1 kN; Spindle runout: < 0.003 mm; Taper contact area: > 85%."
    },
    {
        "manual_id": "MIL-SOP-002",
        "machine_type": "Milling Machine",
        "subsystem": "Axis Ball Screws & Linear Guides",
        "title": "Axis Drive Overcurrent, Ball Screw Backlash, and Guideway Lubrication",
        "failure_modes": [
            "Axis servo motor overload / high power consumption (>30 kW)",
            "Positioning error and backlash (>0.03 mm)",
            "Grinding noise during rapid traverse (G00)",
            "Central lubrication pressure fault"
        ],
        "symptoms": "Power consumption increases disproportionately during linear motion, axis vibration spikes, high motor temp.",
        "root_cause_analysis": "Lubrication distributor metering valve clogged, causing dry friction on linear guideway recirculating ball runners.",
        "sop_steps": [
            "1. Power down and lock servo drive circuit breaker.",
            "2. Check central lube pump pressure gauge during automated cycle (must reach 15 bar within 10 seconds).",
            "3. Inspect progressive distribution block ports and replace blocked metering valves (0.1cc / 0.2cc).",
            "4. Manually grease X/Y/Z ball nuts with Mobilux EP-2 until fresh grease purges past wiper seals.",
            "5. Measure ball screw axial play with dial gauge while jogging axis back and forth.",
            "6. Inspect telescopic way covers for denting and wiper blade damage; clear swarf build-up."
        ],
        "safety_protocol": "Mechanically block the vertical Z-axis column with counterbalance locking pin before servicing ball screw.",
        "required_parts": ["Progressive Lube Valve Block (6-port)", "Way Wiper Seals Kit 35mm", "Linear Rail Ball Runner Block THK HSR-35"],
        "estimated_mttr_hours": 2.5,
        "clearance_tolerances": "Ball screw backlash: < 0.008 mm; Lube cycle pressure: 15 - 20 bar; Axis squareness: < 0.01 mm / 300 mm."
    },

    # ── Industrial Boiler ──────────────────────────────────────────────────────
    {
        "manual_id": "BLR-SOP-001",
        "machine_type": "Industrial Boiler",
        "subsystem": "Feedwater Pump & Combustion Chamber",
        "title": "Feedwater Pump Cavitation, Motor Overheat, and Thermal Imbalance",
        "failure_modes": [
            "Feedwater pump casing temperature >95°C",
            "Severe vertical vibration in multistage pump (>7.0 mm/s)",
            "Drum water level fluctuations",
            "Steam pressure drops below setpoint"
        ],
        "symptoms": "Elevated bearing & motor temperatures (temp_bearing_degC > 90°C), severe vibration (vibration_v_mms > 6.0), erratic feed flow.",
        "root_cause_analysis": "Deaerator tank water temperature too close to boiling causing Net Positive Suction Head (NPSH) breakdown and pump impeller cavitation.",
        "sop_steps": [
            "1. Switch boiler feed control to standby pump unit immediately.",
            "2. Verify deaerator operating pressure (nominal 0.2 - 0.5 bar) and feedwater temperature (102°C - 105°C).",
            "3. Inspect suction strainer for iron oxide scale and boiler sludge deposition.",
            "4. Check mechanical seal flush line for adequate cooling flow (> 4 L/min).",
            "5. Measure multi-stage pump coupling laser alignment (angular < 0.05 mm, offset < 0.05 mm).",
            "6. Inspect pump balancing disk and intermediate diffuser rings for erosion scoring.",
            "7. Conduct boiler water chemical check (Total Dissolved Solids TDS < 3000 ppm, dissolved oxygen < 0.007 mg/L)."
        ],
        "safety_protocol": "HIGH PRESSURE STEAM & THERMAL HAZARD: Wear high-temp arc/steam safety suit, face shield, and insulated gloves. Bleed pressure safely through blowdown valve.",
        "required_parts": ["Multistage Pump Impeller Set (Grundfos CR-45)", "Silicon Carbide Mechanical Seal Cartridge", "Suction Basket Strainer (SS316 40-mesh)"],
        "estimated_mttr_hours": 4.0,
        "clearance_tolerances": "Max bearing vibration: < 3.5 mm/s RMS; Minimum NPSH margin: > 1.8 m; Seal face leakage: < 5 drops/min."
    },
    {
        "manual_id": "BLR-SOP-002",
        "machine_type": "Industrial Boiler",
        "subsystem": "Burner & Forced Draft Fan",
        "title": "Forced Draft (FD) Fan Vibration, Air Damper Actuator, and Burner Flame Instability",
        "failure_modes": [
            "FD fan unbalance vibration (>6.5 mm/s)",
            "Burner flame pulsation / lock-out",
            "High flue gas temp (>220°C)",
            "Motor overheating under load"
        ],
        "symptoms": "Horizontal vibration on FD fan motor increases over 6.0 mm/s, combustion efficiency drops, fluctuating motor load.",
        "root_cause_analysis": "Particulate accumulation on FD fan centrifugal impeller blades creating dynamic rotor unbalance.",
        "sop_steps": [
            "1. Initiate controlled burner purge sequence and lock out fuel gas supply line with double-block-and-bleed valves.",
            "2. Open FD fan housing access hatch and high-pressure steam clean impeller blades.",
            "3. Inspect fan pillow block roller bearings for pitting and grease degradation.",
            "4. Dynamic trim balancing of fan wheel in-situ using vibration analyzer (target ISO 1940 Grade G2.5).",
            "5. Calibrate air-fuel ratio damper servo motor stroke (0 - 90 degrees modulation).",
            "6. Check optical flame scanner sensor lens and clean soot coating."
        ],
        "safety_protocol": "Gas safety shutoff verification mandatory before entry. Perform confined space atmospheric test for CO and O2 levels.",
        "required_parts": ["Pillow Block Bearings SNL-515", "Damper Modulating Actuator Belimo 24V", "Flame Scanner UV Sensor Cell"],
        "estimated_mttr_hours": 3.0,
        "clearance_tolerances": "Fan unbalance: < 2.5 mm/s RMS; Flue gas O2: 3.0% - 4.5%; CO emissions: < 50 ppm."
    },

    # ── Centrifugal Pump ───────────────────────────────────────────────────────
    {
        "manual_id": "PMP-SOP-001",
        "machine_type": "Centrifugal Pump",
        "subsystem": "Impeller, Volute & Mechanical Seal",
        "title": "Centrifugal Pump Cavitation, Mechanical Seal Failure, and Bearing Degradation",
        "failure_modes": [
            "Impeller cavitation (gravel-in-pipe acoustic signature)",
            "Bearing temperature rise (>85°C)",
            "Horizontal vibration spike (>6.5 mm/s)",
            "Seal leakage (> 15 mL/hr)",
            "Discharge pressure decline"
        ],
        "symptoms": "Vibration levels exceed 6.5 mm/s, bearing temperature reaches 85°C+, high power consumption with reduced flow throughput.",
        "root_cause_analysis": "Operating pump far to the right or left of Best Efficiency Point (BEP), suction throttling, or entrained gas causing severe cavitation erosion on impeller vane inlets.",
        "sop_steps": [
            "1. Close discharge isolation valve gradually, then power off pump motor and lock out electrical disconnect switch.",
            "2. Close suction valve and drain pump casing via bottom drain cock.",
            "3. Inspect suction piping for air traps and verify suction pressure gauge reading vs NPSH curve.",
            "4. Disassemble casing split cover and inspect impeller shroud, vanes, and wear rings for honeycombing erosion.",
            "5. Replace dual cartridge mechanical seal (EagleBurgmann Cartex-DN) and flush piping plan 11/53A.",
            "6. Inspect pump shaft sleeve for fretting and runout (< 0.02 mm TIR).",
            "7. Refill bearing housing with synthetic ISO VG 68 turbine oil up to center mark on oil level sight glass.",
            "8. Perform motor-pump laser alignment with thermal growth offset compensation."
        ],
        "safety_protocol": "Hazardous chemical / fluid drainage protocol. Neutralize fluid remnants before casing disassembly. Wear splash goggles.",
        "required_parts": ["Closed Impeller SS316 (Size 250mm)", "Dual Mechanical Seal Cartridge EagleBurgmann", "SKF Explorer Deep Groove Bearings 6312", "Casing Gasket Set"],
        "estimated_mttr_hours": 3.0,
        "clearance_tolerances": "Wear ring radial clearance: 0.25 - 0.35 mm; Shaft runout: < 0.02 mm; Oil level: Mid-sight glass."
    },
    {
        "manual_id": "PMP-SOP-002",
        "machine_type": "Centrifugal Pump",
        "subsystem": "Coupling & Drive Assembly",
        "title": "Pump-to-Motor Angular Misalignment and Resonant Vibration",
        "failure_modes": [
            "2X shaft rotational speed vibration peak",
            "Coupling elastomeric spider element shredding",
            "Motor inboard bearing overheating",
            "Foundation anchor bolt loosening"
        ],
        "symptoms": "Sharp 2x RPM vibration harmonic, motor inboard bearing temperature exceeds 80°C, high vibration on both horizontal and vertical axes.",
        "root_cause_analysis": "Thermal growth differential between cold alignment and operational state, soft foot on pump baseplate, or degraded elastomeric insert.",
        "sop_steps": [
            "1. De-energize motor drive and tag out local disconnect.",
            "2. Remove coupling safety guard and inspect polyurethane spider element for dust or tearing.",
            "3. Check for baseplate soft-foot using dial indicator under each foot while loosening anchor bolts (< 0.05 mm allowed).",
            "4. Shim motor feet with 304 stainless steel precision pre-cut shims.",
            "5. Perform precision laser shaft alignment (Target: Angular misalignment < 0.04 mm/100mm; Offset < 0.05 mm).",
            "6. Retorque baseplate anchor bolts to 140 Nm with Loctite 243 threadlocker.",
            "7. Reinstall coupling guard and conduct full-load baseline vibration spectrum analysis."
        ],
        "safety_protocol": "Never operate without coupling guard securely fastened. Ensure zero rotational coastdown before opening guard.",
        "required_parts": ["Lovejoy / Rotex Polyurethane Coupling Element 98 Shore A", "Precision SS304 Alignment Shim Assortment", "Vibration Isolator Mounts"],
        "estimated_mttr_hours": 2.0,
        "clearance_tolerances": "Angular alignment: < 0.03 deg; Parallel offset: < 0.04 mm; Soft foot: < 0.03 mm."
    }
]


def get_all_manuals() -> list[dict]:
    """Return all equipment manuals for indexing."""
    return EQUIPMENT_MANUALS


def get_manuals_for_machine(machine_type: str) -> list[dict]:
    """Return manuals matching machine type."""
    return [m for m in EQUIPMENT_MANUALS if m["machine_type"].lower() == machine_type.lower()]
