# A Comparison of Simplified Net-Trade and Flow-Based European Power Price Spread Models

*Abstract.* A price difference between two bidding zones arises only when a
transmission limit binds. This study builds an hourly zonal
production-cost model of eight continental European bidding zones. Three models
are compared: a constant annual bound on each
zone's net exports, calculated from past flows (*NTC model*); published hourly bounds (*maxNetPos model*); and published hourly limits on individual network elements (*CNEC model*). Parameters are estimated on 2024 and the models evaluated on 2025.

The models differ in two ways: whether the transfer limit is one number for a
whole country or a separate limit on each monitored line and transformer (network element), and whether it is fixed for the year or recalculated every hour. Getting the number of congested hours right depends mostly on limiting
individual network elements. Getting the response to a gas or carbon price change right
depends significantly on recalculating hourly. None is
better at picking which particular hours are congested; all three are modest.

German and French prices actually differed in 86% of the hours of 2025. The CNEC
model reproduces that closely, at 84%. The NTC model reaches only 53%: in the
other hours it gives the two countries an identical price, so the spread it
reports there is zero. Between Germany and Poland
all three models give much the same answer. All three take the same time to
solve; the CNEC model uses 2.7 times the memory.

*Data.* All inputs are public: EEX auction reports, futures settlements and the
IMF primary commodity database for fuel and carbon prices; the JAO publication
tool for the flow-based domain; the ENTSO-E Transparency Platform for load,
generation, installed capacity, outages and cross-border flows.

---

## 1. Motivation and research question

European day-ahead electricity clears in one daily auction publishing a price
per bidding zone per delivery hour. The *spread* is the difference between two
zones' prices for the same hour.

A non-zero spread requires a binding transmission constraint: two coupled zones
with unconstrained transfer clear at a common price. 

Cross-zonal capacity in the Core region has been calculated flow-based — that
is, by constraining individual network elements rather than bilateral borders —
since 8 June 2022, and on the German–French, German–Dutch, French–Belgian and
Dutch–Belgian borders since 2015 under the smaller region that preceded it
(§4.1). Two questions follow, each asked of the three models set out
in §4:

1. *Reproduction.* Evaluated on 2025, a year none of the model's parameters were
   fitted to, does constraining individual network elements reproduce the
   observed spread distribution more accurately than bounding each zone's net
   exports?
2. *Sensitivity.* Under identical perturbations to the gas price and the
   carbon price, do the three models give the same response?

---

## 2. Price formation in the model

Every power generator offers at short-run marginal cost:

```
SRMC  =  fuel / efficiency  +  CO2 price x emissions / efficiency  +  VOM
```

Nuclear offers €27.18/MWh, lignite around €75, a mid-efficiency CCGT around €85.
The plant where cumulative capacity meets demand sets the price.

In the linear program the price is the *dual variable*. Two zones joined by an unconstrained link share the
same dual, because the optimiser moves power until they do. When a transmission
constraint binds, its own dual is the price difference:

```
mu(constraint)  =  price(importing zone) - price(exporting zone)
```

---

## 3. Model specification

| | |
|---|---|
| Bidding zones | DE_LU, FR, PL, NL, BE, AT, CH, CZ |
| Interconnection | 13 internal links; 19 external borders held at observed net flow |
| Temporal resolution | Hourly — one *snapshot* per delivery hour, 8,784 in 2024 and 8,760 in 2025 |
| Thermal fleet | 865 units, grouped into 115 blocks |
| Problem size | 8 buses, 957 generators, 13 links, 16 loads |
| Solver | HiGHS dual simplex, via linopy and PyPSA; the year solved in 30-day blocks |

The 865 thermal units are grouped into 115 blocks for calculations: units in the same zone, technology and efficiency band have
almost the same marginal cost, so the optimiser does not need them separately.
Availability is still tracked unit by unit.

Reservoir hydro, pumped storage and batteries enter as fixed hourly generation
profiles taken from observed output. They are not optimised, and therefore never
set a price. This is a simplifying assumption; otherwise, optimisation will take far longer and require much more computing resources.

### 3.1 The optimisation problem

All three models are the same linear program and differ only in
constraints.

Indices: zones `z`, snapshots `t`, generators `i`, links `l`. A link has an
orientation, and its flow may take either sign.

*Decision variables.* Generation `g(i,t) >= 0` for each generator and snapshot,
and flow `f(l,t)` for each link and snapshot.

*Objective.* Minimise total generation cost, with `c(i,t)` the short-run
marginal cost of §2:

```
minimise   SUM over i, t of   c(i,t) * g(i,t)
```

*Constraints common to all three models.* For every zone and snapshot, the
energy balance, whose dual variable `lambda(z,t)` is the zonal price:

```
SUM of g(i,t) for generators i in z
  + SUM of f(l,t) for links l entering z
  - SUM of f(l,t) for links l leaving z          =   demand(z,t)
```

For every generator and snapshot, an availability limit; for every link and
snapshot, a rating derived from the preceding week of scheduled exchange
(§4.2); and for every zone and snapshot, an operating-reserve requirement.

*The quantity the models constrain.* A zone's net position is its exports
less its imports across the links it holds:

```
NP(z,t)  =  SUM of f(l,t) for links leaving z
          - SUM of f(l,t) for links entering z
```

and its *Core* net position counts only links whose other end is a Core zone,
adding `x(z,t)`, the observed exchange with the five Core zones the model does
not represent:

```
NPcore(z,t)  =  SUM of f(l,t) for links leaving z to a Core zone
              - SUM of f(l,t) for links entering z from a Core zone
              + x(z,t)
```


---

## 4. The three models

The models are identical in zones, fleet, demand, fuel and carbon prices,
renewable profiles, reserve treatment and acceptance criteria. They differ only
in how cross-zonal transfer is limited.

| model | limit applies to | recomputed hourly | source of the numbers |
|---|---|---|---|
| NTC model (§4.2) | a zone's net position | no, one pair per year | estimated here from observed flows |
| maxNetPos model (§4.4) | a zone's net position | yes | published by JAO |
| CNEC model (§4.3) | individual network elements | yes | published by JAO |

### 4.1 Background

Before a border moved to flow-based coupling — 20 May 2015 for the four Central
Western Europe borders, 8 June 2022 for the rest of Core, and never for the
Swiss borders (Appendix A) — transmission capacity was published for that border
in each direction as a *Net Transfer Capacity*. Three quantities define it.

*Total Transfer Capacity (TTC)* is the largest exchange between two areas that
still respects the network's thermal, voltage and stability limits. 

*Transmission Reliability Margin (TRM)* is a deduction covering the difference
between the forecast used in that simulation and what actually occurs. It is a safety margin.

*Already Allocated Capacity (AAC)* is capacity already sold in earlier auctions
— yearly and monthly products bought before the day-ahead auction opens — and so
no longer available to it. Traders will use however much of it they want. Unallocated capacity, the leftover, is never used.

```
NTC = TTC - TRM
ATC = NTC - AAC
```

*Available Transfer Capacity (ATC)* is what the day-ahead auction can actually
use.

Each border's TTC is conditional on the scenario used to derive it, including
what every *other* border was assumed to be doing. The values on different
borders are therefore not additive.

For each hour JAO publishes both the largest exchange possible
across each individual border, computed on the assumption that every other
zone's net position is zero, and the largest net position each zone may hold
across all its Core borders at once. Adding a zone's border maxima and dividing
by its net-position maximum gives the factor by which independent per-border
limits overstate what that zone can do simultaneously. Averaged over the 24
hours of 1 July 2024 as an example:

| zone | Core borders | sum of border maxima | max net position | overstatement |
|---|---|---|---|---|
| AT | 4 | 13,802 MW | 5,465 MW | **2.55×** |
| BE | 3 | 13,673 MW | 7,165 MW | 1.91× |
| CZ | 4 | 14,387 MW | 7,575 MW | 1.90× |
| DE | 6 | 29,195 MW | 15,998 MW | 1.82× |
| PL | 3 | 7,561 MW | 4,625 MW | 1.63× |
| NL | 2 | 11,496 MW | 7,335 MW | 1.58× |
| FR | 2 | 10,033 MW | 8,307 MW | **1.21×** |

Flow-based capacity calculation constrains individual network elements instead,
so the set of feasible simultaneous exchanges is represented directly. It went
live in the Core capacity calculation region on 8 June 2022. Core comprises the
bidding-zone borders between thirteen EU member states — Austria, Belgium,
Croatia, Czechia, France, Germany, Hungary, Luxembourg, the Netherlands, Poland,
Romania, Slovakia and Slovenia — forming twelve bidding zones, Germany and
Luxembourg being one. Switzerland is not a member, and only Core-to-Core borders
are allocated flow-based.

The Core go-live was not, however, a single transition from net transfer
capacity to flow-based allocation across this footprint, and the study's
borders did not share one history. Flow-based allocation had already been in
use since 20 May 2015 in the smaller *Central Western Europe* region — Germany,
France, Belgium, the Netherlands, and Austria as a separate zone after the
German–Austrian split of 1 October 2018 — so the German–French, German–Dutch,
French–Belgian and Dutch–Belgian borders have been flow-based throughout the
period for which this model has data. At the other extreme, the German–Polish,
German–Czech, Austrian–Czech and Czech–Polish borders were allocated by
*explicit auction* until 17 June 2021, a mechanism in which transmission rights
are sold in a separate auction before the energy auctions clear, so prices need
not converge even when capacity is spare. The three Swiss borders have been
allocated explicitly throughout. Appendix A sets out the regime on each of the
thirteen modelled borders by year.

### 4.2 The NTC model

Two limits apply.

*Per link*, a rating taken from the previous week of trade. Two different
quantities are both called flow across a border: the *commercial exchange*,
which is the volume the market scheduled and which the model's links represent,
and the *physical flow*, which is what the electricity actually did and which
includes power passing through on its way elsewhere. The rating used here is the
largest commercial exchange recorded on that border in the preceding 168 hours,
recalculated every hour. A rolling figure rather than a fixed annual one because
interconnector capacity varies.

*Per zone*, a bound on the net position — that zone's exports less its imports
across all of its borders in the hour.

| zone | lower (MW) | upper (MW) |
|---|---|---|
| DE_LU | −11,777 | 15,154 |
| FR | −8,850 | 11,550 |
| PL | −2,817 | 3,514 |
| NL | −4,046 | 6,924 |
| BE | −6,597 | 2,047 |
| AT | −6,693 | 2,991 |
| CH | −6,821 | 5,047 |
| CZ | −3,167 | 2,987 |

In the notation of §3.1, the constraint added is

```
NPlo(z)  <=  NP(z,t)  <=  NPhi(z)        for every zone z and snapshot t
```

with `NPlo` and `NPhi` taken from the table above. Two rows per zone per
snapshot, sixteen in all, and the bounds have no `t` — the same pair applies in
all 8,760 hours of the year.

Each link keeps its own rating, so no single border is made artificially tight.
What the net-position bound adds is that a zone's borders compete with one
another: Germany may send 5,000 MW to France, but every megawatt sent there is
one it cannot also send to Austria.

That coupling is also a limit inherent to the model. If the bound on
`z` is the only binding constraint in an hour, the dual on it is the value of
one further megawatt of export from `z` *whichever border carries it*, so `z`
is separated from every neighbour it supplies by the same amount. Borders of the
same zone can be separated by different amounts only when a neighbour's own
bound binds, or a link reaches its rating. A single limit per zone cannot
express a constraint that affects two of that zone's borders differently.

The bounds are the 0.5th and 99.5th percentiles of each zone's observed hourly
net position, multiplied by 1.05. The percentiles discard extremes. The 5% provides a small but significant flex.

This study does not use published NTC
values, and none exist for the study period. Day-ahead net transfer capacities
ceased to be published for the German–French, German–Dutch, French–Belgian and
Dutch–Belgian borders when Central Western Europe adopted flow-based allocation
in 2015, and for the remaining Core borders at the 2022 go-live (Appendix A).
The bound estimated here is a substitute. Because it is estimated from the flows
that actually occurred in these years, it is likely to fit them better than a
published NTC series would have.

### 4.3 The CNEC model

A *critical network element with contingency (CNEC)* is a monitored network
element — typically a line or transformer. For each CNEC and
each hour the domain publishes two quantities.

A *power transfer distribution factor (PTDF)* for every Core zone: the change in
that element's loading, in MW, per MW increase in that zone's net position. 

A *remaining available margin (RAM)*: how much loading is still available on
that element for the day-ahead auction to use. It is the element's thermal
rating less two deductions — the *reference flow*, which is the loading the TSOs
expect on it before the auction clears given forecast generation, load and
already-scheduled exchange; and a *reliability margin* covering the difference
between that forecast and what actually occurs.

```
RAM  =  thermal rating  -  reference flow  -  reliability margin
```

In the notation of §3.1, the constraint is

```
SUM over Core zones z of  PTDF(z,e,t) * NPcore(z,t)   <=   RAM(e,t) + s(e,t)
```

for every monitored element `e` and every snapshot `t`, where `s(e,t) >= 0` is
the slack variable described below. Five Core zones are unrepresented in the
model (Croatia, Hungary, Romania, Slovakia, Slovenia); their contribution
enters through `x(z,t)` inside `NPcore`.

### 4.4 The maxNetPos model

It bounds a zone's net position, as the NTC model does, but uses
JAO's published `maxNetPos` series — the maximum and minimum net position each
Core zone may hold, recomputed for every hour.

It therefore serves two purposes. Against the NTC model it isolates the value of
hourly variation alone. Against the CNEC model it isolates the value of
constraining individual elements alone. 

The constraint has the same form as §4.2 but is written on the Core net
position and has a right-hand side that varies by snapshot:

```
NPmin(z,t)  <=  NPcore(z,t)  <=  NPmax(z,t)      for z in Core, every t
```

where `NPmin` and `NPmax` are JAO's published minimum and maximum net position
for zone `z` in hour `t`. The row count is unchanged from §4.2 — two per Core
zone per snapshot, plus two for Switzerland — so the difference from the NTC
model lies entirely in the right-hand side.

### 4.5 The three models side by side

Zones, fleet, demand, fuel and carbon prices, renewable profiles, link ratings,
reserve treatment, bid ladder and acceptance criteria are common to all three.
What differs is the constraint.

| | NTC model | maxNetPos model | CNEC model |
|---|---|---|---|
| quantity bounded | `NP(z,t)`, a zone's total net position | `NPcore(z,t)`, its net position across Core borders | loading on one network element under one contingency |
| coefficients in a row | 1 on `z`, zero on every other zone | 1 on `z`, zero on every other zone | a separate `PTDF(z,e,t)` on each of the seven modelled Core zones |
| rows per snapshot | 16 | 16 | 105–140 |
| rows per year | 140,160 | 140,160 | 1,151,640 (§8.4) |
| right-hand side varies by hour | no | yes | yes |
| right-hand side estimated here | yes | no | no |
| source | 0.5th and 99.5th percentiles of the observed net position, × 1.05 | JAO `maxNetPos` | JAO `PTDF` and `RAM` |
| Switzerland | bounded like any zone | estimated bound, being outside Core | estimated bound, being outside Core |
| two borders of one zone separable by different amounts | no | no | yes |
| slack variables | none | none | one per row, €5,000/MW |

*What each costs to assemble.* One script of 9.4 KB builds the NTC bounds into a
144-byte file. The `maxNetPos` series arrives from the same fetcher as the
domain and needs only a column rename, 1.8 MB for the year. The flow-based path
took fourteen scripts and holds roughly 176 MB for the evaluation year: 183 raw
domain downloads (≈119 MB), a 33 MB consolidated table, 17 MB of JAO's own
record of which constraints were binding, used to verify the reconstruction,
and the
net-position and scheduled-exchange series the scope correction requires. 

---

## 5. Estimation and evaluation design

Parameters are estimated on *2024* and the models evaluated on *2025*.

2023 is excluded on substantive grounds. German nuclear ran until April 2023 and
French nuclear was recovering from the 2022 stress-corrosion outages, so
evaluating on it would mix fleet change with model error.

2026 is excluded on comparability. The year is incomplete: roughly eight months have been delivered at the time of writing, and every acceptance criterion in §5.2 is computed on an annual distribution of hourly prices. 

### 5.1 The nuclear bid ladder

Downloaded data nuclear SRMCs have no variability, but this is unlikely to be true. The study assumes a nuclear operator holding capacity back for a more valuable hour will not offer it at fuel cost. So, while average costs may drop due to scale economies, marginal costs are assumed to rise. A rising offer curve approximates that, and the nuclear 'bid ladder' reflects that.

An example with French nuclear. The value of `s` was chosen by running the model on 2024 at values from €0 to
€40/MWh and comparing against the acceptance criteria. It was set to *€20/MWh* based on research.

### 5.2 How each parameter was set

| parameter | year used | method |
|---|---|---|
| `bid_ladder.Nuclear` = 20 | 2024 | swept 0–40 |
| `must_run.Nuclear` | 2024 | matched to counts of negative-price hours |
| net-position source rule | 2024 | measured |
| reserve multipliers 3.0 / 3.0 | 2024 | swept; default not improved on |
| `negative_bidding` depths | 2024 | matched to counts of negative-price hours |
| net-position bounds | 2024–2026 | flow percentiles — *includes the evaluation year* |
| `scarcity_tiers` price steps | 2025 | set on the evaluation year |

---

## 6. Reproduction of the observed spread distribution

Fifteen acceptance criteria were written before any parameter was estimated.
They cover two borders and three zones and measure four things: the average
price level in each zone, the mean and standard deviation of each spread, the
10th and 90th percentiles of each zonal price distribution, and two measures of
hour-by-hour agreement between the modelled and observed spread series.

### 6.1 Estimation year (2024)

| criterion | NTC model | maxNetPos model | CNEC model |
|---|---|---|---|
| DE mean price | −18.3% ❌ | −17.8% ❌ | **−13.6% ✅** |
| FR mean price | −17.4% ❌ | −20.0% ❌ | −27.9% ❌ |
| PL mean price | −13.8% ✅ | −14.0% ✅ | −6.4% ✅ |
| DE-FR mean spread | −4.25 ✅ | **−2.40 ✅** | +5.46 ❌ |
| DE-FR spread sd | 0.76× ✅ | 0.66× ❌ | **1.08× ✅** |
| DE-FR sign agreement | 69% / 10% ❌ | 71% / 10% ❌ | 81% / 50% ❌ |
| DE-FR hourly correlation | 0.44 ❌ | 0.54 ✅ | **0.56 ✅** |
| DE-PL mean spread | −1.05 ✅ | **−0.52 ✅** | −4.54 ✅ |
| DE-PL spread sd | **1.00× ✅** | 0.81× ✅ | 1.41× ❌ |
| DE-PL sign agreement | 11% / 77% ❌ | 8% / 74% ❌ | 41% / 86% ❌ |
| DE-PL hourly correlation | 0.29 ❌ | 0.33 ❌ | 0.28 ❌ |
| DE p90 | 104 vs 125 ✅ | 105 vs 125 ✅ | 107 vs 125 ✅ |
| DE p10 | 0 vs 10 ❌ | 0 vs 10 ❌ | −7 vs 10 ❌ |
| PL p90 | 108 vs 142 ✅ | 108 vs 142 ✅ | 112 vs 142 ✅ |
| PL p10 | 31 vs 50 ❌ | 30 vs 50 ❌ | **39 vs 50 ✅** |
| **met** | **7 of 15** | **7 of 15** | **8 of 15** |


### 6.2 Evaluation year (2025)

| criterion | NTC model | maxNetPos model | CNEC model |
|---|---|---|---|
| DE mean price | −17.4% ❌ | −16.0% ❌ | **−14.1% ✅** |
| FR mean price | −3.4% ✅ | −11.5% ✅ | −18.6% ❌ |
| PL mean price | −14.0% ✅ | −13.9% ✅ | −9.0% ✅ |
| DE-FR mean spread | −13.47 ❌ | −7.27 ❌ | **−1.21 ✅** |
| DE-FR spread sd | 0.61× ❌ | 0.69× ❌ | **0.90× ✅** |
| DE-FR sign agreement | 66% / 14% ❌ | 77% / 15% ❌ | 84% / 52% ❌ |
| DE-FR hourly correlation | 0.59 ✅ | **0.63 ✅** | 0.58 ✅ |
| DE-PL mean spread | −1.00 ✅ | **+0.24 ✅** | −3.17 ✅ |
| DE-PL spread sd | 0.82× ✅ | 0.80× ✅ | **1.00× ✅** |
| DE-PL sign agreement | 13% / 72% ❌ | 12% / 67% ❌ | 44% / 87% ❌ |
| DE-PL hourly correlation | 0.53 ✅ | 0.52 ✅ | 0.52 ✅ |
| DE p90 | 115 vs 141 ✅ | 116 vs 141 ✅ | 120 vs 141 ✅ |
| DE p10 | −0 vs 4 ❌ | −0 vs 4 ❌ | −3 vs 4 ❌ |
| PL p90 | 115 vs 155 ❌ | **116 vs 155 ✅** | **116 vs 155 ✅** |
| PL p10 | 41 vs 41 ✅ | 40 vs 41 ✅ | 74 vs 41 ❌ |
| **met** | **8 of 15** | **9 of 15** | **10 of 15** |

### 6.3 How often each border was constrained

Two zones that can trade freely clear at the same price, so a price difference
between them means transfer between them was limited in that hour. Counting the
hours where the price difference exceeds €0.50/MWh therefore measures how often
each border was constrained, using published prices alone and no transmission
data of any kind.

*2025*

| border | NTC model | maxNetPos model | CNEC model | observed |
|---|---|---|---|---|
| DE-FR | 53.1% | 62.0% | **84.4%** | 85.5% |
| DE-PL | 61.5% | 57.4% | **85.5%** | 83.2% |
| DE-CZ | 36.2% | 34.4% | 83.1% | 76.3% |
| DE-NL | 17.1% | 15.7% | 82.4% | 74.3% |
| FR-BE | 33.4% | 52.8% | 85.2% | 82.6% |
| DE-AT | 9.1% | 9.9% | 73.6% | 82.2% |

Across the thirteen modelled borders in 2025, observed prices differ in 65% to
97% of hours. The NTC model produces a difference in 9% to 62%. On most borders it
therefore leaves transfer unlimited: nothing binds, the two zones
clear at the same price, and no spread arises at all. The CNEC model produces a
difference in 74% to 86% of hours.

*The 2025 table answers the attribution question the maxNetPos model was built
to settle.* On DE–FR the NTC model falls 32.4 percentage points short of
observed.
Replacing its estimated constant bound with JAO's published hourly bound
recovers 8.9 of those points; replacing the per-zone bound with per-element
constraints recovers 31.3. Roughly a quarter of the improvement is attributable
to hourly variation and the remainder to constraining individual elements. On
DE–PL, DE–CZ and DE–NL the published hourly bound is no better than the
estimated constant one and on two of the three slightly worse, while the CNEC
model improves all of them by 20 to 65 points.


### 6.4 Spread distributions

*2024*

| | mean | median | sd |
|---|---|---|---|
| DE-FR, NTC model | 16.25 | 0.64 | 28.91 |
| DE-FR, CNEC model | 25.95 | 26.43 | 41.02 |
| DE-FR, observed | 20.49 | 5.04 | 37.95 |
| DE-PL, NTC model | −18.79 | −8.07 | 37.39 |
| DE-PL, CNEC model | −22.28 | −10.31 | 52.80 |
| DE-PL, observed | −17.74 | −8.06 | 37.46 |

*2025*

| | mean | median | sd |
|---|---|---|---|
| DE-FR, NTC model | 14.78 | **0.00** | 22.52 |
| DE-FR, maxNetPos model | 20.98 | — | 25.67 |
| DE-FR, CNEC model | 27.03 | 35.60 | 33.28 |
| DE-FR, observed | 28.25 | 17.77 | 37.05 |
| DE-PL, NTC model | −15.97 | −4.38 | 25.57 |
| DE-PL, maxNetPos model | −14.73 | — | 24.92 |
| DE-PL, CNEC model | −18.14 | −5.84 | 31.20 |
| DE-PL, observed | −14.97 | −3.39 | 31.27 |

The DE–FR mean rises at each step across the three models — 14.78,
20.98, 27.03 against 28.25 observed — as does its standard deviation. On DE–PL
the maxNetPos model has the smallest mean error of the three (+0.24 against
observed), while the CNEC model has much the best standard deviation (1.00 times
observed against 0.80).

*The NTC model's DE–FR median in 2025 is exactly zero.* German and French prices
are identical in more than half of all hours, against an observed median of
€17.77/MWh. This is the failure described in §1: where the transfer limit does
not bind, the two zones clear at one price and there is no spread to measure.

*The CNEC model matches how widely DE–PL prices vary in 2025 but not in 2024* —
a standard deviation of 31.20 against 31.27 observed in 2025, but 52.80 against 37.46 in
1.    The difference is scarcity pricing. The 2024 CNEC runs produce 8 to 19
hours per run with a price above €500/MWh; the 2025 runs produce none. A small
number of very high prices inflates the standard deviation without moving the
median much.

*No model matches the shape of the DE–FR distribution.* The observed
spread is close to zero in most hours with occasional large values, which is why
its median (17.77) sits well below its mean (28.25). The CNEC model's median
(35.60) sits *above* its mean (27.03): it produces a moderate spread in most
hours instead of a small one in most hours and a large one in a few. The NTC
model produces no spread at all in most hours.

---

## 7. Perturbation experiment

### 7.1 Design

Nine runs per model — a base case and four drivers at plus and minus one
standard deviation. All three models were run on the evaluation year,
twenty-seven runs; the estimation year has the NTC and CNEC models only,
eighteen. Both directions are run for each driver, because the response need not
be the same size in each: a large enough gas price movement puts combined-cycle
gas above lignite in the merit order and changes which technology is marginal.

Each perturbation alters one input series. Fleet, demand, network, transmission
limits, reserve, bid ladder and criteria are held identical across all
forty-five runs.

| driver | perturbation | effect |
|---|---|---|
| Gas | ±1 sd added to every day of the TTF series | moves gas offers; the change is divided by each plant's efficiency, so it also spreads the gas plants further apart |
| Carbon | ±1 sd added to the EUA series | moves lignite most, gas less, nuclear and renewables not at all, so it changes the order of the stack rather than shifting it |
| Weather | wind and solar capacity factors × (1 ± rel) | changes how much renewable output there is, leaving its hour-to-hour pattern intact |
| FR nuclear | French nuclear availability × (1 ± rel) | scales how much of the fleet can run, keeping its seasonal outage pattern |

Each model is compared against *its own* base case:

```
response(model)  =  spread(shock, model) - spread(base, model)
finding            =  response(CNEC) - response(NTC)
                   =  [ response(maxNetPos) - response(NTC)       ]   hourly step
                    + [ response(CNEC)      - response(maxNetPos) ]   element step
```

### 7.2 Perturbation sizes

```
gas          23.9%    1 sd of the DAILY series, 2024-2026
carbon       10.6%    1 sd of the DAILY series
weather       1.4%    1 sd of the ANNUAL mean, n = 2
FR nuclear    0.63%   1 sd of the ANNUAL mean, n = 2
```

The four sizes do not measure the same thing. Gas and carbon are one standard
deviation of the daily price series, so they describe how far the price
actually moves from day to day. Weather and French nuclear are one standard
deviation of an annual mean taken over two years, which is the difference
between 2024 and 2025 and nothing more. French nuclear availability was 0.668
in 2024 and 0.674 in 2025, giving 0.63%, against a real range an order of
magnitude larger: France produced roughly 279 TWh in 2022 during the
stress-corrosion outages and 360 TWh in 2024.

A perturbation of 1.4% or 0.63% moves the spread by a few hundredths of a euro,
and any ratio built on it divides two quantities too small to tell apart. The
DE–FR weather ratio was the largest number the experiment produced and the
least informative one.

*Weather and French nuclear are therefore not reported in §8.* Both were run,
in both directions, for all three models, and the runs are in the repository.
What is missing is a defensible perturbation size, not the results. §10 lists
the longer history that would supply one.

---

## 8. Results

### 8.1 Sensitivity ratio, NTC model against CNEC model

*Sensitivity* is half the difference between the two responses,
`(response(+1sd) − response(−1sd)) / 2`. Using both directions this way cancels
the base case and measures how steeply the spread changes with the driver. The
*ratio* is the CNEC model's
sensitivity divided by the NTC model's.

| border | driver | 2024 | 2025 |
|---|---|---|---|
| DE-FR | gas | 1.73 | 2.65 |
| DE-FR | carbon | 1.34 | 1.52 |
| DE-PL | gas | 1.22 | 1.17 |
| DE-PL | carbon | 1.01 | 1.01 |

*On DE–PL the two agree* — carbon at 1.01 in both years, gas at 1.22 and 1.17.
The bound estimated here is adequate for this border.

*On DE–FR they do not* — carbon 1.34 then 1.52, gas 1.73 then 2.65, the same
direction in both years. The NTC model gives a response a third to a half
smaller than the CNEC model's.

### 8.2 Which step accounts for the sensitivity difference

| border | driver | NTC | maxNetPos | CNEC | total gap | from hourly | from elements |
|---|---|---|---|---|---|---|---|
| DE-FR | gas | 0.90 | 1.98 | 2.40 | 1.49 | **1.08** | 0.41 |
| DE-FR | carbon | 1.87 | 2.35 | 2.84 | 0.97 | 0.48 | 0.49 |
| DE-PL | gas | 1.32 | 1.33 | 1.55 | 0.23 | 0.00 | 0.23 |
| DE-PL | carbon | −1.81 | −1.66 | −1.83 | −0.02 | 0.15 | −0.17 |

Reproducing how *often* a border separates
requires the constraint to bind in the right hours, which depends on which
element binds; a per-zone bound can bind in approximately the right number of
hours while being wrong about which ones. The *size* of the price response to a
fuel-price perturbation depends instead on how tight the limit is in the hours
where the perturbation moves the merit order, and a bound held constant across
the year cannot tighten in those hours whether it is written on a zone or on an
element. 

### 8.3 Driver elasticities

Change in spread, in €/MWh, per 100% change in the driver. CNEC model.

| driver | DE-FR 2024 | DE-FR 2025 | DE-PL 2024 | DE-PL 2025 |
|---|---|---|---|---|
| carbon | 25.0 | 26.8 | 19.5 | 17.2 |
| gas | 12.3 | 10.0 | 7.0 | 6.5 |

Carbon's elasticity is two to three times gas's, on both borders and in both
years. Gas looks the larger driver in the raw responses only because its
perturbation is 23.9% against carbon's 10.6%.

---

## 9. Conclusions

*It helps on one of the two borders tested and not the other.* On DE–PL the
three models give sensitivities within 0.23 €/MWh of each other, and the NTC and
CNEC models agree in both years, so the published data adds nothing there. On
DE–FR the CNEC model responds 1.3 to 2.7 times more strongly than the NTC model,
in the same direction in both years. The flow-based domain is therefore not
better in general. It is better on some borders. Saying in advance which borders
would need a mechanism this study does not establish.

*A per-zone limit does not shrink the spread. It removes it.* On 2025 the NTC
model's DE–FR spread has a median of exactly zero: German and French prices come
out identical in more than half of all 8,760 hours, against an observed median
of €17.77/MWh. Anyone using such a model to value an interconnector, or a
contract settling on the DE–FR difference, would read zero in most hours. That
is a different kind of error from being wrong by a few euros.

*Which feature of the published data matters depends on the question asked.*
The NTC and CNEC models differ in two ways at once, so a third model was built
that is per zone like the first and hourly like the second (§4.4). Splitting the
difference in two gives opposite answers to the study's two questions.

For how often a border is congested, what matters is limiting elements rather
than zones. On DE–FR the NTC model falls 32.4 percentage points short of the
observed frequency; making the per-zone limit hourly recovers 8.9 of them, and
limiting elements recovers 31.3 (§6.3).

For how strongly the spread responds to a price change, what matters is that the
limit is recomputed hourly. On the same border and year, the hourly step accounts
for 1.08 of a 1.49 gap on gas, and half the gap on carbon (§8.2).

## 10. Further work

This study fixes every input at its observed value and changes one at a time.
Four extensions would matter more than the rest.

*Monte Carlo simulation.* Perturbing one driver at a time measures the response
to each in isolation. It cannot say how often a wide spread occurs, or what the
distribution of outcomes looks like when several drivers move together, which is
what anyone valuing an interconnector or a spread contract needs. Drawing fuel
prices, carbon prices, weather and plant availability jointly, and solving the
year many times over, would turn a set of sensitivities into a distribution.
The constraint representation would matter more in that setting, not less: the
per-zone bound is estimated from one observed pattern of flows, so the further a
draw moves from that pattern, the less the bound describes anything real.

*Energy storage.* Reservoir hydro, pumped storage and batteries are all held at
their observed output and never set a price. They share a property the rest of
the fleet does not: they move energy between hours rather than producing it, so
the price at which an operator is willing to discharge is the value of the hour
being given up, not the cost of fuel. A model that cannot represent that has
nothing offering in the middle of the cost range, which is why the nuclear bid
ladder is being asked to compensate. Letting storage optimise against that
opportunity cost — for reservoir hydro, the water value, the price below which an
operator would rather keep the water than generate — is the route to the
price-level errors and to the shape of the DE–FR distribution. It also matters
more each year, as storage takes a growing share of the flexible capacity these
zones rely on.

*More countries and more years.* Eight zones and two years is the smallest
credible version of this experiment. Extending the footprint would test whether
the border-by-border result holds anywhere beyond DE–FR and DE–PL. Extending the
history would give perturbation sizes drawn from a real distribution rather than
the difference between two adjacent years, and would let the model be evaluated
across conditions — a fuel-price crisis, a nuclear outage year — that the two
years used here do not contain.

*Forecasting.* Everything here is retrospective. The flow-based domain is
published two days ahead and exists for no future date, so a forward-looking
model needs a constructed substitute, and the accuracy that substitute costs has
to be measured before it is trusted. The per-zone bound has the opposite
property: it is easy to assume forward, and this study shows what an unchanging
bound costs in sensitivity. Establishing how each behaves out of sample is the
step between a model that reproduces the past and one that says anything about
the future.

---

## Appendix A. Allocation regime by border, 2015–2026

How day-ahead cross-zonal capacity was allocated on each of the thirteen
modelled borders. Three regimes appear.

*Flow-based (FB)* — implicit allocation, limits placed on individual network
elements. The energy auction and the flow are decided together.

*Net transfer capacity (NTC)* — implicit allocation, a single limit per border.
The energy auction and the flow are still decided together; only the form of
the limit differs.

*Explicit auction (EXP)* — transmission rights are sold in a separate auction
before the energy auctions clear. A trader must forecast the price difference
in advance, so capacity may go unused in hours when using it would have paid,
and prices need not converge even when the border is not full.

| border | 2015 | 2016 | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DE–FR | FB¹ | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB |
| DE–NL | FB¹ | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB |
| FR–BE | FB¹ | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB |
| NL–BE | FB¹ | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB | FB |
| DE–AT | — | — | — | FB² | FB | FB | FB | FB | FB | FB | FB | FB |
| DE–BE | — | — | — | — | — | FB³ | FB | FB | FB | FB | FB | FB |
| DE–CZ | EXP | EXP | EXP | EXP | EXP | EXP | NTC⁴ | FB⁵ | FB | FB | FB | FB |
| DE–PL | EXP | EXP | EXP | EXP | EXP | EXP | NTC⁴ | FB⁵ | FB | FB | FB | FB |
| AT–CZ | EXP | EXP | EXP | EXP | EXP | EXP | NTC⁴ | FB⁵ | FB | FB | FB | FB |
| CZ–PL | EXP | EXP | EXP | EXP | EXP | EXP | NTC⁴ | FB⁵ | FB | FB | FB | FB |
| DE–CH | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP |
| FR–CH | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP |
| AT–CH | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP | EXP |

¹ from 20 May 2015, the Central Western Europe flow-based go-live; implicit NTC
coupling before that
² from 1 October 2018 — not a border before the German–Austrian bidding zone
split, Germany and Austria having formed a single zone
³ from 18 November 2020 — no interconnector before, ALEGrO being the first
electrical link between the two countries
⁴ from 17 June 2021, when Interim Coupling replaced explicit auctions on
PL–DE, PL–CZ, PL–SK, CZ–DE, CZ–AT and HU–AT
⁵ from 8 June 2022, the Core flow-based go-live, which also closed the Central
Western Europe scheme
