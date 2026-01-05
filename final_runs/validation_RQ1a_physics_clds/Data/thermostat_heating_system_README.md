# Thermostat Heating System CLD

## Purpose
This CLD represents a classic thermostat-controlled home heating system based on well-established physics principles. It is designed to validate the judge system's functionality on a domain with:
- Clear, uncontested causal relationships
- Strong literature support (physics textbooks)
- Simple, well-understood dynamics

## Physics Principles

### Newton's Law of Cooling
Heat transfer rate is proportional to the temperature difference:
```
Q = hA(T_room - T_outside)
```
Where h is the heat transfer coefficient and A is the surface area.

### First Law of Thermodynamics (Energy Balance)
The rate of temperature change depends on heat input and output:
```
dT_room/dt = (Q_heating - Q_loss) / (m * c_p)
```
Where m is the air mass and c_p is specific heat capacity.

### Feedback Control
The thermostat creates a negative feedback loop that regulates room temperature:
- Temperature Gap = Desired Temperature - Room Temperature
- Heating activates when gap is positive (room too cold)
- The system oscillates around the setpoint

## Variables (7 total)

| Variable | Type | Description |
|----------|------|-------------|
| Room Temperature | Stock | Current air temperature in the room |
| Desired Temperature | Exogenous | Thermostat setpoint |
| Temperature Gap | Auxiliary | Desired - Actual temperature |
| Heating Rate | Flow | Rate of heat addition from furnace |
| Heat Loss Rate | Flow | Rate of heat escaping to outside |
| Outside Temperature | Exogenous | Ambient outdoor temperature |
| Thermal Resistance | Parameter | Building insulation quality (R-value) |

## Edges (8 total)

| Source | Target | Polarity | Physics Basis |
|--------|--------|----------|---------------|
| Room Temperature | Temperature Gap | NEGATIVE | Gap = Desired - Room |
| Desired Temperature | Temperature Gap | POSITIVE | Gap = Desired - Room |
| Temperature Gap | Heating Rate | POSITIVE | Thermostat control logic |
| Heating Rate | Room Temperature | POSITIVE | First law of thermodynamics |
| Room Temperature | Heat Loss Rate | POSITIVE | Newton's law of cooling |
| Outside Temperature | Heat Loss Rate | NEGATIVE | Reduces ΔT |
| Heat Loss Rate | Room Temperature | NEGATIVE | First law (heat leaves) |
| Thermal Resistance | Heat Loss Rate | NEGATIVE | Q = ΔT/R |

## Key References

1. **Sterman, J.D. (2000)**. Business Dynamics: Systems Thinking and Modeling for a Complex World. McGraw-Hill.
   - Chapters 5-6: Thermostat dynamics as canonical feedback example

2. **Incropera, F.P. et al. (2007)**. Fundamentals of Heat and Mass Transfer. 6th Edition. Wiley.
   - Newton's law of cooling, thermal resistance

3. **Çengel, Y.A. & Boles, M.A. (2015)**. Thermodynamics: An Engineering Approach. 8th Edition. McGraw-Hill.
   - First law of thermodynamics, energy balance

4. **ASHRAE (2017)**. ASHRAE Handbook - Fundamentals.
   - Building heat loss calculations, R-values

5. **Åström, K.J. & Murray, R.M. (2008)**. Feedback Systems: An Introduction for Scientists and Engineers. Princeton University Press.
   - Feedback control theory, PID control

## Expected Judge Results
- All edges should receive HIGH judge scores (CORRECT verdicts)
- Literature support should be easily found in physics databases
- This establishes baseline performance for the judge on "easy" domains









