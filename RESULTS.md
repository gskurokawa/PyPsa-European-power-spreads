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
model reproduces that closely, at 84%. The NTC model reaches only 53%, so in the remaining
47% of hours it gives the two countries the same price and reports no spread
at all. Between Germany and Poland all three models give much the same
answer. All three take the same time to
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

*Relation to existing work.* Comparing transmission representations inside one
model is not new: Gunkel et al. (2020) set a flow-based representation against
net transfer capacities in the Balmorel model and found that flow-based
modelling flattens prices across space, while noting that the spatial clustering
their model required left the size of the difference uncertain. Most such work
constructs its own flow-based domain, and the construction is not neutral —
Schönheit et al. (2021) show that the choice of monitored network elements and
the size of the reliability margin change the resulting domain more than the
generation shift keys do, and Weinhold (2021) builds a domain precisely in order
to test one of its policy parameters. A comparison resting on a constructed
domain therefore mixes the effect of the representation with the effect of the
assumptions behind it. A separate, empirical literature measures what flow-based
allocation did to the market rather than to a model; Ovaere et al. (2023) find
that cross-border exchange volumes and price convergence both rose after the
2015 go-live in Central Western Europe. This study instead takes JAO's published
hourly domain as given, imposes it and the two alternatives on an otherwise
identical model built in PyPSA (Brown et al., 2018), and compares three
representations rather than two — which separates two features that a straight
flow-based-against-NTC test holds together: whether the limit is recomputed
every hour, and whether it is written on a zone or on a network element.

---

## 2. Price formation in the model

Every power generator offers at short-run marginal cost:

$$\text{SRMC} \;=\; \frac{\text{fuel}}{\eta} \;+\; \frac{p_{\mathrm{CO_2}} \cdot \varepsilon}{\eta} \;+\; \text{VOM}$$

with $\eta$ the thermal efficiency, $p_{\mathrm{CO_2}}$ the carbon price and $\varepsilon$ the emissions intensity of the fuel.

Nuclear offers €27.18/MWh, lignite around €75, a mid-efficiency CCGT around €85.
The plant where cumulative capacity meets demand sets the price.

In the linear program the price is the *dual variable*. Two zones joined by an unconstrained link share the
same dual, because the optimiser moves power until they do. When a transmission
constraint binds, its own dual is the price difference:

$$\mu_{\text{constraint}} \;=\; \lambda_{\text{importing}} \;-\; \lambda_{\text{exporting}}$$

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

*Why these eight.* Seven of them — DE_LU, FR, PL, NL, BE, AT and CZ — are Core
zones, so their borders are the ones the published flow-based domain actually
constrains, which is what this study tests. Switzerland is not a Core member and
its borders are not allocated flow-based, but it is included because it sits in
the middle of the synchronous area and carries a large share of the transit
between its neighbours; leaving it out would push that flow onto a frozen
external border where it could not respond to anything. The five remaining Core
zones — Croatia, Hungary, Romania, Slovakia and Slovenia — are peripheral to the
borders under test, and enter instead through their observed net positions, as
§4.3 describes. Eight zones is also what keeps the flow-based model inside the
memory of a single desktop machine.

The 865 thermal units are grouped into 115 blocks for calculations: units in the same zone, technology and efficiency band have
almost the same marginal cost, so the optimiser does not need them separately.
Availability is still tracked unit by unit.

Reservoir hydro and pumped storage enter as fixed hourly generation profiles
taken from observed output, at zero marginal cost. They are not optimised, and
therefore never set a price. For pumped storage only the generation is included;
the consumption while pumping is not, so the model holds a small quantity of
energy that was never bought. Batteries are absent altogether, since installed
capacity and storage duration are not published on the same basis as the rest of
the fleet. These are simplifying assumptions. Optimising storage would require
each hour to be solved together with the hours around it, which is far more
expensive than solving them separately, and §10 sets out what it would cost and
what it would change.

### 3.1 The optimisation problem

All three models are the same linear program and differ only in
constraints.

Indices: zones $z$, snapshots $t$, generators $i$, links $l$. A link has an
orientation, and its flow may take either sign.

*Decision variables.* Generation $g_{i,t} \ge 0$ for each generator and snapshot,
and flow $f_{l,t}$ for each link and snapshot.

*Objective.* Minimise total generation cost, with $c_{i,t}$ the short-run
marginal cost of §2:

$$\min \sum_{i}\sum_{t} c_{i,t}\, g_{i,t}$$

*Constraints common to all three models.* For every zone and snapshot, the
energy balance, whose dual variable $\lambda_{z,t}$ is the zonal price:

$$\sum_{i \in z} g_{i,t} \;+\; \sum_{l \,\to\, z} f_{l,t} \;-\; \sum_{l \,\leftarrow\, z} f_{l,t} \;=\; d_{z,t}$$

where $l \to z$ are the links entering zone $z$ and $l \leftarrow z$ those leaving it.

For every generator and snapshot, an availability limit; for every link and
snapshot, a rating derived from the preceding week of scheduled exchange
(§4.2); and for every zone and snapshot, an operating-reserve requirement.

*The quantity the models constrain.* A zone's net position is its exports
less its imports across the links it holds:

$$NP_{z,t} \;=\; \sum_{l \,\leftarrow\, z} f_{l,t} \;-\; \sum_{l \,\to\, z} f_{l,t}$$

and its *Core* net position counts only links whose other end is a Core zone,
adding $x_{z,t}$, the observed exchange with the five Core zones the model does
not represent:

$$NP^{\text{core}}_{z,t} \;=\; \sum_{l \,\leftarrow\, z,\; l \in C} f_{l,t} \;-\; \sum_{l \,\to\, z,\; l \in C} f_{l,t} \;+\; x_{z,t}$$

where $C$ is the set of links whose other end is a Core zone.


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

$$NTC = TTC - TRM \qquad\qquad ATC = NTC - AAC$$

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

$$NP^{\text{lo}}_{z} \;\le\; NP_{z,t} \;\le\; NP^{\text{hi}}_{z} \qquad \forall\, z,\; \forall\, t$$

with $NP^{\text{lo}}$ and $NP^{\text{hi}}$ taken from the table above. Two rows per zone per
snapshot, sixteen in all, and the bounds have no $t$ — the same pair applies in
all 8,760 hours of the year.

Each link keeps its own rating, so no single border is made artificially tight.
What the net-position bound adds is that a zone's borders compete with one
another: Germany may send 5,000 MW to France, but every megawatt sent there is
one it cannot also send to Austria.

That coupling is also a limit inherent to the model. If the bound on
$z$ is the only binding constraint in an hour, the dual on it is the value of
one further megawatt of export from $z$ *whichever border carries it*, so $z$
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

$$RAM \;=\; F^{\max} \;-\; F_{0} \;-\; FRM$$

with $F^{\max}$ the thermal rating, $F_{0}$ the reference flow and $FRM$ the flow reliability margin.

In the notation of §3.1, the constraint is

$$\sum_{z \in \text{Core}} PTDF_{z,e,t}\; NP^{\text{core}}_{z,t} \;\le\; RAM_{e,t} + s_{e,t} \qquad \forall\, e,\; \forall\, t, \qquad s_{e,t} \ge 0$$

for every monitored element $e$ and every snapshot $t$, where $s_{e,t} \ge 0$ is
the slack variable described below. Five Core zones are unrepresented in the
model (Croatia, Hungary, Romania, Slovakia, Slovenia); their contribution
enters through $x_{z,t}$ inside $NP^{\text{core}}$.

### 4.4 The maxNetPos model

It bounds a zone's net position, as the NTC model does, but uses
JAO's published `maxNetPos` series — the maximum and minimum net position each
Core zone may hold, recomputed for every hour.

It therefore serves two purposes. Against the NTC model it isolates the value of
hourly variation alone. Against the CNEC model it isolates the value of
constraining individual elements alone. 

The constraint has the same form as §4.2 but is written on the Core net
position and has a right-hand side that varies by snapshot:

$$NP^{\min}_{z,t} \;\le\; NP^{\text{core}}_{z,t} \;\le\; NP^{\max}_{z,t} \qquad \forall\, z \in \text{Core},\; \forall\, t$$

where $NP^{\min}$ and $NP^{\max}$ are JAO's published minimum and maximum net position
for zone $z$ in hour $t$. The row count is unchanged from §4.2 — two per Core
zone per snapshot, plus two for Switzerland — so the difference from the NTC
model lies entirely in the right-hand side.

### 4.5 The three models side by side

Zones, fleet, demand, fuel and carbon prices, renewable profiles, link ratings,
reserve treatment, bid ladder and acceptance criteria are common to all three.
What differs is the constraint.

| | NTC model | maxNetPos model | CNEC model |
|---|---|---|---|
| quantity bounded | $NP_{z,t}$, a zone's total net position | $NP^{\text{core}}_{z,t}$, its net position across Core borders | loading on one network element under one contingency |
| coefficients in a row | 1 on $z$, zero on every other zone | 1 on $z$, zero on every other zone | a separate $PTDF_{z,e,t}$ on each of the seven modelled Core zones |
| rows per snapshot | 16 | 16 | 105–140 |
| rows per year | 140,160 | 140,160 | 1,151,640 (§8.4) |
| right-hand side varies by hour | no | yes | yes |
| right-hand side estimated here | yes | no | no |
| source | 0.5th and 99.5th percentiles of the observed net position, × 1.05 | JAO `maxNetPos` | JAO $PTDF$ and $RAM$ |
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

An example with French nuclear. The value of $s$ was chosen by running the model on 2024 at values from €0 to
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

*Separating as often as the market is not the same as separating in the same
hours.* Every figure above counts hours; none of them asks whether the hours
coincide. For that, take the two series as indicators — 1 in an hour where the
border separated, 0 otherwise, one series for the model and one for the market —
and correlate them. On binary data the Pearson correlation is the *phi
coefficient*, computed from the four cell counts of the 2×2 table: $a$ hours
where both separated, $b$ model only, $c$ market only, $d$ neither, summing
to $N$.

The derivation is short. Writing $X$ and $Y$ for the two indicators, $XY = 1$
only when both are 1, so $E[XY] = a/N$, while $E[X] = (a+b)/N$ and
$E[Y] = (a+c)/N$. Hence

$$\operatorname{Cov}(X,Y) \;=\; \frac{a}{N} - \frac{(a+b)(a+c)}{N^{2}} \;=\; \frac{ad - bc}{N^{2}}$$

the $a^{2}$, $ab$ and $ac$ terms cancelling. A 0/1 variable satisfies $X^{2} = X$,
so $\operatorname{Var}(X) = E[X] - E[X]^{2} = (a+b)(c+d)/N^{2}$, and likewise
$\operatorname{Var}(Y) = (a+c)(b+d)/N^{2}$. Dividing, the $N^{2}$ cancels:

$$\phi \;=\; \frac{ad - bc}{\sqrt{(a+b)(c+d)(a+c)(b+d)}}$$

It is zero exactly when $ad = bc$, the condition for the two series to be
independent, and unlike a hit rate it cannot be raised by separating in every
hour, because such a model has $c = d = 0$ and no variance to correlate.

| border | NTC model | maxNetPos model | CNEC model |
|---|---|---|---|
| DE-FR | 0.261 | **0.295** | 0.230 |
| DE-PL | 0.258 | **0.265** | 0.244 |
| FR-BE | 0.221 | **0.320** | 0.291 |
| DE-CZ | 0.159 | 0.169 | **0.222** |
| DE-NL | 0.049 | 0.031 | **0.219** |
| DE-AT | −0.013 | −0.018 | **0.148** |

Two readings follow, and they pull in opposite directions.

*Where the per-zone models barely separate at all, the flow-based domain adds
real timing information.* On DE–AT, DE–NL and DE–CZ — separating in 9%, 17% and
36% of hours under the NTC model — phi rises from about zero to 0.15–0.22.

*Where the per-zone models already separate often, it adds none.* On DE–FR,
DE–PL and FR–BE the CNEC model's phi is no higher than the maxNetPos model's and
on the first two lower than the NTC model's, despite separating in 84% of hours
against 53%. The gain in phi therefore tracks the gain in *frequency* rather
than any improvement in timing, which is consistent with the model reaching the
right number of congested hours without reaching the right ones.

No value exceeds 0.32. On the three Swiss borders, where every model keeps an
estimated bound, the hit rate equals its random benchmark to the first decimal
and phi is within 0.05 of zero: the modelled congestion there is statistically
independent of the observed congestion.

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

The NTC model's DE–FR median prints as 0.00 against €17.77/MWh observed. This
is not a separate result. §6.3 already reports that the model separates the
border in 53.1% of hours, so it shows no meaningful difference in the other
47%, and the median falls inside that block because a small share of hours
carry a negative spread. Had that share been a point or two smaller the median
would print as a small positive number instead, with nothing about the model
changed. The separation frequency is the robust statement; the median is that
frequency seen through a statistic sensitive to an unrelated detail.

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
model produces no meaningful spread in 47% of hours.

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

$$R_m \;=\; S_m(\text{shock}) - S_m(\text{base})$$

$$\underbrace{R_{\text{CNEC}} - R_{\text{NTC}}}_{\text{total}} \;=\; \underbrace{R_{\text{maxNetPos}} - R_{\text{NTC}}}_{\text{hourly step}} \;+\; \underbrace{R_{\text{CNEC}} - R_{\text{maxNetPos}}}_{\text{element step}}$$

where $S_m$ is the mean spread under model $m$.

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
$\tfrac{1}{2}\left[R(+1\sigma) - R(-1\sigma)\right]$. Using both directions this way cancels
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

*Why the CNEC model responds most.* A spread appears when two zones want to
trade more than the network will carry. Raising the gas price makes Germany
more expensive, so more zones want to buy from Germany's neighbours, and the
desired flow across the border grows. Whether that produces a price difference
depends on whether the limit stops the flow. The limit in the CNEC model is
recomputed every hour and written on the network elements that actually carry
the power, so it is tight in the hours the system is stressed — which are the
same hours a fuel-price change moves the merit order. The limit in the NTC
model is one number for the whole year. It is an average, so it is too loose
in the tight hours and too tight in the slack ones, and it cannot tighten in
response to anything. The maxNetPos model sits between the two: it is
recomputed every hour, so it tightens when the system is stressed, but it is
written on a zone's total position rather than on the elements, so it cannot
distinguish one of that zone's borders from another. The three sensitivities
order themselves accordingly.

### 8.3 Driver elasticities

Change in spread, in €/MWh, per 100% change in the driver. CNEC model.

| driver | DE-FR 2024 | DE-FR 2025 | DE-PL 2024 | DE-PL 2025 |
|---|---|---|---|---|
| carbon | 25.0 | 26.8 | 19.5 | 17.2 |
| gas | 12.3 | 10.0 | 7.0 | 6.5 |

Carbon's elasticity is two to three times gas's, on both borders and in both
years. Gas looks the larger driver in the raw responses only because its
perturbation is 23.9% against carbon's 10.6%.

### 8.4 Abbreviated Monte Carlo study on a twelve-day sample

This is an abbreviated Monte Carlo study where gas and carbon prices are randomised to observe how power price speads behave in each of the NTC, maxNetPos, and CNEC models. It is abbreviated in terms of the sample size to avoid computational resource inadequecies.

*Scope.* This section reports two borders, DE–FR and DE–PL,
and summarises the other eleven in the appendix. Mean spreads and other figures are computed over 288
hours — a sample: one Wednesday of each month, so twelve days of 2025, whereas previous sections calculated over 8760 hours. 

*The sample.* The second Wednesday of each month of 2025. December moves to the third Wednesday because three hours of
French load are absent from the ENTSO-E series on 10 December.
All twelve are working days, which carry higher demand than weekends.

*What was varied.* The drivers: the *gas price*, in euros per MWh thermal, and the
*carbon price*, in euros per tonne. One draw is a pair of price *additions* or *shifts*, added
to the gas and carbon prices in each hour of each of the twelve days. The coal price and every other fuel cost are held at
their observed values.

The gas and carbon price addition pair is drawn from a bivariate normal. Its standard deviations are those of
the daily gas and carbon prices — €9.14/MWh thermal for gas and €7.54/t for
carbon, over the 974 daily observations the repository holds, 1 January 2024 to
31 August 2026. Its correlation is 0.222, measured on *weekly changes* in the two series. 

Each of the NTC, maxNetPos, and CNEC models' power price spread set was averaged over the 288 hours of 2025 (one run), 100 times - as 100 pairs of gas and carbon price additions were drawn. 

#### Does the abbreviated study recover §8.2's sensitivities?

Each model's mean spread over the twelve days was regressed on the gas and carbon price shifts divided by its own standard deviation above, so the coefficients are
euros per MWh spread per one standard deviation of driver. 

€/MWh of spread per one standard deviation of the driver:

| | | §8.2 gas | §8.4 gas | §8.2 carbon | §8.4 carbon |
|---|---|---|---|---|---|
| DE–FR | NTC | 0.90 | 1.14 | 1.87 | 1.98 |
| | maxNetPos | 1.98 | 2.04 | 2.35 | 2.72 |
| | CNEC | 2.40 | 2.41 | 2.84 | 3.04 |
| DE–PL | NTC | 1.32 | 2.34 | −1.81 | −1.25 |
| | maxNetPos | 1.33 | 1.99 | −1.66 | −1.14 |
| | CNEC | 1.55 | 2.70 | −1.83 | −1.52 |

Both borders rank the three models as §8.2 does, and DE–PL keeps its negative
carbon coefficient: a higher carbon price lifts coal-heavy Poland more than
gas-heavy Germany, which narrows the difference between them. The ratio between
the outer two models is close on both borders — on DE–PL, 2.70 against 2.34 here
and 1.55 against 1.32 in §8.2.

#### Is the spread a straight-line function of the two drivers?

Each regression reports an $R^{2}$: the share of the variation across the 100
runs that the two draws account for. On DE–FR it is 0.986, 0.998 and 0.992; on
DE–PL, 0.891, 0.905 and 0.979. Over the range drawn — −2.42 to +2.41 standard
deviations of gas, −2.75 to +2.31 of carbon — the mean spread is close
to a straight-line function of the two.

That matters because a straight-line relationship can be calculated instead of
simulated. If the mean spread moves by $a$ euros per standard deviation of gas
and $b$ per standard deviation of carbon, and the two drivers have correlation
$\rho$, the standard deviation of the resulting spreads is

$$\sqrt{a^{2} + b^{2} + 2ab\rho}$$

Evaluating that with each model's own coefficients and $\rho$, and
comparing it with how much the 100 mean spreads actually varied, the two agree
to the second decimal place on 37 of the 39 border-and-model combinations. 

---

## 9. Conclusions

*The published flow-based data helps on one of the two borders tested and not
the other.* On DE–PL the
three models give sensitivities within 0.23 €/MWh of each other, and the NTC and
CNEC models agree in both years, so the published data adds nothing there. On
DE–FR the CNEC model responds 1.3 to 2.7 times more strongly than the NTC model,
in the same direction in both years. The flow-based domain is therefore not
better in general. It is better on some borders. Saying in advance which borders
would need a mechanism this study does not establish.

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

*Energy storage.* Storage moves energy between hours rather than producing it,
so the price at which an operator is willing to discharge is the value of the
hour being given up, not the cost of fuel. This model represents none of that.
Storage is either absent or held at its observed output at zero cost, so nothing
offers in the middle of the cost range, which is part of why the nuclear bid
ladder of §5.1 is being asked to compensate. The three cases differ, and are
worth separating.

*Batteries.* Neither their consumption while charging nor
their output while discharging enters the model. The reason is data: ENTSO-E's
production types have no battery category, so installed capacity is not
published on the same basis as the rest of the fleet, and the storage duration
each unit holds is not published at all. Both would have to come from a source
outside the public series this study restricts itself to. Even with the data,
the addition would not be free. A battery couples one hour to the next through
its state of charge, and it is the absence of any such coupling that makes the
30-day solution blocks of §3 exact rather than approximate. The omission is not
neutral: a battery adds supply in the expensive hours and demand in the cheap
ones, which reduces how much a zone needs to trade, so the spreads reported here
are likely to be wider than they would be with batteries present.

*Pumped storage.* ENTSO-E reports both the
generation and the consumption of these plants; this model keeps the first and
discards the second, and prices the output at zero. The model therefore contains
energy that was never bought. The quantity is small — pumped storage is a low
single-digit share of generation in these zones — so the effect on the spreads
is unlikely to be large, but its direction is known. The free energy arrives in
the peak hours, where it relieves tightness that the market actually faced, so
it reduces how often a border binds. Adding the pumping consumption as a load
would remove that free energy and should widen the spreads slightly. That
correction is cheap, because the consumption series is already downloaded. The
remaining two errors are not: pricing the output at its true opportunity cost,
and letting the plant choose its own hours rather than replaying observed ones,
means representing it as a storage unit, which introduces the inter-hour
coupling described above and removes the exact 30-day decomposition.

*Reservoir hydro.* Its input is inflow, which is
free, so there is no purchase missing from the model and no energy created that
did not exist. What is wrong is the price. Observed generation enters at zero
marginal cost, so water is always inframarginal and never sets a price, when in
much of this footprint — Austria and Switzerland in particular — it regularly
does. The correct offer is the water value: the price below which an operator
would rather keep the water than generate. Assigning one as a fixed cost is
computationally free but requires a number this study has no public basis for
choosing. Deriving it inside the model, by letting the reservoir optimise across
the year, is the version that would blow up the computational requirement, since
a seasonal reservoir cycles once a year and cannot be represented inside a
30-day block at all. That would require the whole year to be solved as a single
problem, which is roughly twelve times the size of the largest problem solved
here.

*More countries and more years.* Eight zones and two years is the smallest
credible version of this experiment. Extending the footprint would test whether
the border-by-border result holds anywhere beyond DE–FR and DE–PL. Extending the
history would give perturbation sizes drawn from a real distribution rather than
the difference between two adjacent years, and would let the model be evaluated
across conditions — a fuel-price crisis, a nuclear outage year — that the two
years used here do not contain.

*Monte Carlo simulation.* A more comprehensive Monte Carlo, instead of being limited to the 288 hours of 2025 in this study. Other drivers could be shocked as well. 

*Forecasting.* Forecasting studies are left for possible future work. Running the models in this study forward means
have future inputs, and the three models differ sharply in how hard that is. The CNEC model needs the flow-based domain, and JAO does not publish a domain
for a delivery day until the day before it. For any date further ahead there is
nothing to read, so a forward-looking CNEC model would have to predict the
domain: the PTDF coefficients and the remaining margin on each of roughly 120
monitored elements, for every hour. The NTC model needs nothing further. Its bound is one pair of numbers per zone,
estimated from past flows and held constant, so it can be carried to any future
date unchanged. The maxNetPos model sits between them. JAO publishes its series day-ahead as well, so it has the CNEC
model's problem, but the object to be predicted is two numbers per zone per hour
rather than a domain over 120 elements. 


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

---

## Appendix B. Monte Carlo results for all thirteen borders

### B.1 Mean spread, €/MWh

| border | model | observed | mean | gas | carbon | $R^{2}$ | sd |
|---|---|---|---|---|---|---|---|
| DE–FR | NTC | 34.52 | 17.00 | 1.14 | 1.98 | 0.986 | 2.58 |
|  | maxNetPos |  | 24.42 | 2.04 | 2.72 | 0.998 | 3.85 |
|  | CNEC |  | 30.61 | 2.41 | 3.04 | 0.992 | 4.42 |
| DE–PL | NTC | -8.66 | -9.81 | 2.34 | -1.25 | 0.891 | 2.39 |
|  | maxNetPos |  | -9.48 | 1.99 | -1.14 | 0.905 | 2.04 |
|  | CNEC |  | -14.19 | 2.70 | -1.52 | 0.979 | 2.65 |
| DE–BE | NTC | 7.87 | 7.00 | -0.39 | 0.90 | 0.987 | 0.86 |
|  | maxNetPos |  | 2.98 | 0.00 | 0.45 | 0.957 | 0.46 |
|  | CNEC |  | 5.49 | -1.87 | 1.44 | 0.971 | 1.98 |
| DE–NL | NTC | 2.98 | 4.30 | -0.46 | 0.48 | 0.928 | 0.57 |
|  | maxNetPos |  | 3.41 | -0.17 | 0.36 | 0.955 | 0.35 |
|  | CNEC |  | 3.56 | -1.12 | 0.74 | 0.979 | 1.13 |
| DE–CZ | NTC | -3.36 | -3.11 | 1.72 | -0.71 | 0.991 | 1.63 |
|  | maxNetPos |  | -4.85 | 1.21 | -0.55 | 0.956 | 1.17 |
|  | CNEC |  | -9.38 | 1.17 | -0.90 | 0.976 | 1.23 |
| DE–AT | NTC | -6.75 | 3.77 | -0.39 | 0.30 | 0.880 | 0.43 |
|  | maxNetPos |  | 3.20 | -0.30 | 0.34 | 0.865 | 0.40 |
|  | CNEC |  | 9.52 | 0.74 | 0.61 | 0.942 | 1.11 |
| DE–CH | NTC | -5.62 | 4.94 | 0.17 | 0.61 | 0.926 | 0.70 |
|  | maxNetPos |  | 9.07 | 0.88 | 0.90 | 0.979 | 1.45 |
|  | CNEC |  | 13.32 | 1.15 | 1.18 | 0.956 | 1.92 |
| FR–BE | NTC | -26.65 | -10.00 | -1.52 | -1.08 | 0.989 | 2.12 |
|  | maxNetPos |  | -21.44 | -2.04 | -2.27 | 0.996 | 3.48 |
|  | CNEC |  | -25.13 | -4.28 | -1.61 | 0.983 | 5.02 |
| FR–CH | NTC | -40.14 | -12.07 | -0.97 | -1.37 | 0.992 | 1.91 |
|  | maxNetPos |  | -15.35 | -1.16 | -1.82 | 0.995 | 2.43 |
|  | CNEC |  | -17.29 | -1.27 | -1.86 | 0.998 | 2.54 |
| NL–BE | NTC | 4.89 | 2.70 | 0.07 | 0.42 | 0.976 | 0.45 |
|  | maxNetPos |  | -0.44 | 0.17 | 0.09 | 0.924 | 0.23 |
|  | CNEC |  | 1.93 | -0.75 | 0.70 | 0.948 | 0.87 |
| AT–CH | NTC | 1.13 | 1.17 | 0.56 | 0.31 | 0.989 | 0.71 |
|  | maxNetPos |  | 5.87 | 1.19 | 0.56 | 0.996 | 1.45 |
|  | CNEC |  | 3.80 | 0.41 | 0.58 | 0.952 | 0.82 |
| AT–CZ | NTC | 3.39 | -6.88 | 2.11 | -1.01 | 0.979 | 2.04 |
|  | maxNetPos |  | -8.05 | 1.51 | -0.89 | 0.937 | 1.53 |
|  | CNEC |  | -18.90 | 0.43 | -1.51 | 0.936 | 1.47 |
| CZ–PL | NTC | -5.30 | -6.71 | 0.62 | -0.54 | 0.523 | 0.94 |
|  | maxNetPos |  | -4.63 | 0.78 | -0.59 | 0.808 | 0.90 |
|  | CNEC |  | -4.81 | 1.53 | -0.62 | 0.971 | 1.47 |

### B.2 Share of hours the border separated

A border counts as separated in an hour when the two zones' prices differ by
more than €0.50/MWh, as in §6.3. Coefficients are percentage points per one
standard deviation of the driver.

| border | model | observed | mean | gas | carbon | $R^{2}$ |
|---|---|---|---|---|---|---|
| DE–FR | NTC | 88.2% | 58.6% | -5.81 | +1.71 | 0.959 |
|  | maxNetPos |  | 71.5% | -2.27 | +0.60 | 0.889 |
|  | CNEC |  | 88.1% | +0.08 | +0.00 | 0.011 |
| DE–PL | NTC | 86.5% | 53.6% | -9.88 | +2.21 | 0.894 |
|  | maxNetPos |  | 46.6% | -7.81 | +1.75 | 0.845 |
|  | CNEC |  | 88.3% | +0.76 | +0.44 | 0.236 |
| DE–BE | NTC | 84.4% | 30.6% | -3.67 | +1.06 | 0.885 |
|  | maxNetPos |  | 14.6% | -1.52 | +0.38 | 0.694 |
|  | CNEC |  | 88.2% | -0.80 | +0.45 | 0.385 |
| DE–NL | NTC | 78.5% | 20.5% | -2.25 | +0.77 | 0.768 |
|  | maxNetPos |  | 14.7% | -0.74 | +0.39 | 0.448 |
|  | CNEC |  | 82.4% | -1.59 | +0.84 | 0.759 |
| DE–CZ | NTC | 77.8% | 31.0% | -3.69 | +0.86 | 0.905 |
|  | maxNetPos |  | 26.4% | -4.06 | +0.73 | 0.894 |
|  | CNEC |  | 87.7% | +1.30 | +0.47 | 0.460 |
| DE–AT | NTC | 79.9% | 13.5% | -1.48 | +0.44 | 0.771 |
|  | maxNetPos |  | 12.9% | -0.75 | +0.14 | 0.511 |
|  | CNEC |  | 78.7% | +2.49 | -0.27 | 0.856 |
| DE–CH | NTC | 94.4% | 23.7% | -1.27 | +0.35 | 0.860 |
|  | maxNetPos |  | 27.9% | -0.30 | +0.05 | 0.464 |
|  | CNEC |  | 53.6% | +1.39 | -0.22 | 0.740 |
| FR–BE | NTC | 85.4% | 42.2% | -3.26 | +0.95 | 0.846 |
|  | maxNetPos |  | 65.1% | -1.49 | +0.46 | 0.768 |
|  | CNEC |  | 89.0% | -0.22 | +0.08 | 0.101 |
| FR–CH | NTC | 98.6% | 36.8% | -4.83 | +1.40 | 0.955 |
|  | maxNetPos |  | 46.0% | -2.09 | +0.61 | 0.846 |
|  | CNEC |  | 48.2% | -1.55 | +0.36 | 0.884 |
| NL–BE | NTC | 74.7% | 23.1% | -2.79 | +0.84 | 0.959 |
|  | maxNetPos |  | 11.0% | -1.33 | +0.35 | 0.825 |
|  | CNEC |  | 87.0% | -0.92 | +0.56 | 0.682 |
| AT–CH | NTC | 96.2% | 20.3% | -1.28 | +0.30 | 0.779 |
|  | maxNetPos |  | 22.8% | +0.03 | -0.04 | 0.022 |
|  | CNEC |  | 52.4% | +1.42 | -0.17 | 0.785 |
| AT–CZ | NTC | 79.5% | 38.6% | -6.14 | +1.62 | 0.929 |
|  | maxNetPos |  | 32.7% | -6.76 | +1.46 | 0.973 |
|  | CNEC |  | 90.1% | +0.31 | +0.58 | 0.260 |
| CZ–PL | NTC | 78.5% | 45.8% | -6.24 | +1.41 | 0.859 |
|  | maxNetPos |  | 34.4% | -4.91 | +1.21 | 0.818 |
|  | CNEC |  | 71.7% | -0.62 | +1.67 | 0.274 |

---

## References

Brown, T., Hörsch, J. and Schlachtberger, D. (2018). PyPSA: Python for Power
System Analysis. *Journal of Open Research Software*, 6(4).
https://doi.org/10.5334/jors.188

Gunkel, P. A., Koduvere, H., Kirkerud, J. G., Fausto, F. J. and Ravn, H. V.
(2020). Modelling transmission systems in energy system analysis: A comparative
study. *Journal of Environmental Management*, 262, 110289.
https://www.sciencedirect.com/science/article/abs/pii/S0301479720302243

Ovaere, M., Kenis, M., Van den Bergh, K., Bruninx, K. and Delarue, E. (2023).
The effect of flow-based market coupling on cross-border exchange volumes and
price convergence in Central Western European electricity markets. *Energy
Economics*, 118, 106519.
https://www.sciencedirect.com/science/article/abs/pii/S0140988323000178

Schönheit, D., Kenis, M., Lorenz, L., Möst, D., Delarue, E. and Bruninx, K.
(2021). Toward a fundamental understanding of flow-based market coupling for
cross-border electricity trading. *Advances in Applied Energy*, 2, 100027.
https://www.sciencedirect.com/science/article/pii/S2666792421000202

Weinhold, R. (2021). Evaluating Policy Implications on the Restrictiveness of
Flow-based Market Coupling with High Shares of Intermittent Generation: A Case
Study for Central Western Europe. arXiv:2109.04940.
https://arxiv.org/abs/2109.04940
