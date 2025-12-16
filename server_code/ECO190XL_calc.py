import anvil.server
import numpy as np
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import griddata

# ======== SFP-beregning ========
def p_sfp800(x):  return -0.0183 * x**2 + 4.5205 * x - 209.04
def p_sfp1000(x): return -0.0107 * x**2 + 2.5426 * x - 37.726
def p_sfp1200(x): return -0.0134 * x**2 + 3.6287 * x - 94.402
def p_sfp1500(x): return -0.012  * x**2 + 3.2252 * x - 3.5191
def p_sfp1620(x): return -0.0071 * x**2 + 1.6626 * x + 159.8

POLY_LIST = [
    (800,  p_sfp800),
    (1000, p_sfp1000),
    (1200, p_sfp1200),
    (1500, p_sfp1500),
    (1620, p_sfp1620),
]

def ekstrapoler_under(pressure, low_point, next_point):
  sfp_i, p_i = low_point; sfp_j, p_j = next_point
  frac = (pressure - p_i) / (p_j - p_i)
  return sfp_i + frac * (sfp_j - sfp_i)

def ekstrapoler_over(pressure, prev_point, high_point):
  sfp_i, p_i = prev_point; sfp_j, p_j = high_point
  frac = (pressure - p_i) / (p_j - p_i)
  return sfp_i + frac * (sfp_j - sfp_i)

def interpolate_sfp(flow, pressure):
  lst = [(sfp, poly(flow)) for sfp, poly in POLY_LIST]
  lst.sort(key=lambda x: x[1])
  # lineær interpolation eller ekstrapolation
  for i in range(len(lst)-1):
    sfp_i, p_i = lst[i]; sfp_j, p_j = lst[i+1]
    if p_i <= pressure <= p_j:
      frac = (pressure - p_i) / (p_j - p_i)
      return sfp_i + frac*(sfp_j - sfp_i)
  # udenfor interval
  if pressure < lst[0][1]:
    return ekstrapoler_under(pressure, lst[0], lst[1])
  else:
    return ekstrapoler_over(pressure, lst[-2], lst[-1])

# ======== Varmegenvinding ========
# Data: [flow m3/h, Udetemp °C, eff %, T_afkast °C, T_supply °C]
hr_data = np.array([
    [100,  0, 84.9,  6.0, 17.0],
    [200,  0, 81.4,  6.6, 16.3],
    [300,  0, 79.4,  6.8, 15.9],
    [100,  5, 84.7,  7.3, 17.7],
    [200,  5, 81.3,  7.8, 17.2],
    [300,  5, 79.3,  8.1, 16.9],
    [100, 10, 84.6, 11.5, 18.5],
    [200, 10, 81.1, 11.9, 18.1],
    [300, 10, 79.2, 12.1, 17.9],
    [100, 15, 84.4, 15.8, 19.2],
    [200, 15, 81.0, 15.9, 19.0],
    [300, 15, 79.0, 16.0, 19.0],
])
points_hr   = hr_data[:, :2]
eff_hr      = hr_data[:, 2]
T_afkast_hr = hr_data[:, 3]
T_supply_hr = hr_data[:, 4]
cp_air      = 1.005  # kJ/(kg·K)
rho_air     = 1.2    # kg/m³

def _interp_heat(values, flow, T_out):
  # Robust interpolation/ekstrapolation for varmegenvinding (gæt op til flow=700)
  flow = float(min(float(flow), 700.0))
  T_out = float(T_out)

  temps = np.unique(points_hr[:,1].astype(float))
  temps.sort()

  # Først: værdi som funktion af flow for hvert temp-niveau
  vals_t = []
  for t in temps:
    m = np.isclose(points_hr[:,1].astype(float), t)
    f = points_hr[m][:,0].astype(float)
    v = values[m].astype(float)
    order = np.argsort(f)
    f = f[order]; v = v[order]
    if len(f) == 0:
      vals_t.append(np.nan)
      continue
    if len(f) == 1:
      vals_t.append(float(v[0]))
      continue
    if flow <= f[0]:
      vals_t.append(float(v[0] + (v[1]-v[0])/(f[1]-f[0])*(flow-f[0])))
    elif flow >= f[-1]:
      vals_t.append(float(v[-1] + (v[-1]-v[-2])/(f[-1]-f[-2])*(flow-f[-1])))
    else:
      vals_t.append(float(np.interp(flow, f, v)))

  vals_t = np.array(vals_t, dtype=float)

  # Dernæst: interpolation/ekstrapolation i udetemperatur
  if len(temps) == 1:
    return float(vals_t[0])
  if T_out <= temps[0]:
    return float(vals_t[0] + (vals_t[1]-vals_t[0])/(temps[1]-temps[0])*(T_out-temps[0]))
  if T_out >= temps[-1]:
    return float(vals_t[-1] + (vals_t[-1]-vals_t[-2])/(temps[-1]-temps[-2])*(T_out-temps[-1]))
  return float(np.interp(T_out, temps, vals_t))

def interpolate_heatrecovery(flow, T_out):
  eff = _interp_heat(eff_hr,      flow, T_out)
  aft = _interp_heat(T_afkast_hr, flow, T_out)
  sup = _interp_heat(T_supply_hr, flow, T_out)
  return eff, aft, sup

def calculate_power(flow, T_ude, T_supply):
  mass_flow = (flow/3600) * rho_air  # kg/s
  return mass_flow * cp_air * (T_ude - T_supply)  # kW

# ======== LYD-DATA ========
# Hver række: [flow, pressure, L63...L8000, Total]
# (samme data som tidligere)
lw_supply = np.array([
    [126,  70, 19.8,24.4,37.1,39.2,31.9,24.9,23.1,18.5,44.9],
    [126, 100,22.4,30.1,41.2,42.2,33.8,27.1,24.4,17.4,45.5],
    [162,  70,20.9,28.1,40.2,45.1,35.6,32.3,29.9,23.6,47.4],
    [162, 100,22.1,30.2,41.2,46.2,38.1,33.3,30.8,25.3,48.2],
    [216,  70,23.0,29.8,40.8,48.9,38.6,37.4,37.5,30.6,50.6],
    [216, 100,26.9,31.4,41.4,48.9,41.1,38.9,38.3,31.8,50.9],
    [250, 150,27.8,34.4,44.3,50.8,43.8,41.7,41.4,36.8,53.8],
    [250, 200,31.9,36.3,45.2,51.3,45.4,43.9,42.6,37.1,54.5],
])
lw_extract = np.array([
    [126,  70,19.0,26.4,28.1,28.5,27.0,24.2,20.2,16.6,35.4],
    [126, 100,20.7,27.6,28.9,30.0,27.7,24.8,20.7,16.7,37.7],
    [162,  70,21.0,27.6,28.6,30.9,27.4,27.5,23.4,19.2,36.6],
    [162, 100,20.5,26.6,28.7,31.8,27.7,27.2,23.2,19.1,37.3],
    [216,  70,20.5,25.4,31.1,34.8,33.4,31.6,28.6,22.6,40.4],
    [216, 100,21.9,27.1,31.5,38.0,35.7,32.4,29.4,24.2,42.6],
    [250, 150,23.3,27.3,33.4,40.3,37.6,35.4,32.4,25.9,44.6],
    [250, 200,23.6,29.5,34.8,41.7,38.4,36.6,33.2,26.1,45.9],
])
lw_outside = np.array([
    [126,  70,16.6,19.8,26.7,25.4,24.7,22.6,19.6,16.8,32.2],
    [126, 100,17.4,20.9,28.9,27.0,26.5,22.7,19.2,16.3,33.6],
    [162,  70,15.4,20.3,26.2,27.1,25.6,24.8,22.3,18.1,33.2],
    [162, 100,21.4,20.6,27.6,29.0,27.9,25.8,22.9,18.9,35.0],
    [216,  70,19.7,21.1,27.7,31.5,29.9,29.4,27.6,21.5,37.0],
    [216, 100,20.1,21.5,30.1,32.2,32.3,29.6,28.0,23.1,38.4],
    [250, 150,21.0,23.9,32.4,36.3,37.4,35.2,34.6,28.1,42.6],
    [250, 200,18.4,24.0,34.3,37.3,38.9,34.6,32.4,21.0,43.4],
])
lw_exhaust = np.array([
    [126,  70,20.5,29.9,40.9,42.7,39.6,33.4,30.8,25.5,47.7],
    [126, 100,20.7,31.2,42.1,43.4,40.4,34.0,30.4,23.9,48.6],
    [162,  70,23.2,30.4,41.5,46.0,41.9,38.1,36.7,32.6,50.2],
    [162, 100,23.5,31.3,42.1,46.7,43.7,38.3,36.5,33.9,51.1],
    [216,  70,25.2,33.5,43.8,49.5,45.9,43.0,43.5,39.9,54.3],
    [216, 100,27.2,33.8,44.5,51.9,46.9,44.4,43.8,40.4,55.7],
    [250, 150,31.1,36.6,46.5,52.6,50.4,47.5,47.1,43.7,57.8],
    [250, 200,31.6,37.0,46.4,54.0,51.5,47.8,47.5,44.1,58.9],
])
lp_cabinet = np.array([
    [126,  70,13.8,23.2,29.3,30.8,29.4,30.0,24.0,20.0,40.8],
    [126, 100,14.2,23.6,30.5,32.1,30.4,30.1,23.4,18.5,42.2],
    [162,  70,13.5,23.0,30.6,33.8,32.1,35.4,29.1,25.3,44.3],
    [162, 100,12.5,25.7,30.5,37.8,33.9,35.8,29.9,25.4,45.5],
    [216,  70,12.4,25.2,31.7,40.2,37.9,42.2,35.8,31.2,49.4],
    [216, 100,14.5,26.5,32.4,39.8,37.3,40.7,33.3,32.7,49.8],
    [250, 150,19.9,28.8,35.0,41.8,41.8,44.0,42.3,37.6,51.9],
    [250, 200,20.6,28.7,35.7,42.1,41.4,43.5,40.0,37.1,53.0],
])

# Robust 2-trins interpolation/ekstrapolation (flow -> tryk)
# Formål: undgå broadcast-fejl når tryk-trin har forskelligt antal flow-punkter,
# og kunne "gætte" værdier op til 700 m3/h og 500 Pa.
MAX_FLOW = 700.0
MAX_PRESSURE = 500.0

def _interp1_extrap(x, xp, fp):
  xp = np.asarray(xp, dtype=float)
  fp = np.asarray(fp, dtype=float)
  order = np.argsort(xp)
  xp = xp[order]
  fp = fp[order]
  x = float(x)
  if len(xp) == 0:
    return np.nan
  if len(xp) == 1:
    return float(fp[0])
  if x <= xp[0]:
    x1, x2 = xp[0], xp[1]
    y1, y2 = fp[0], fp[1]
    return float(y1 + (y2 - y1) / (x2 - x1) * (x - x1))
  if x >= xp[-1]:
    x1, x2 = xp[-2], xp[-1]
    y1, y2 = fp[-2], fp[-1]
    return float(y2 + (y2 - y1) / (x2 - x1) * (x - x2))
  return float(np.interp(x, xp, fp))

def _interp_flow_at_pressure(port_data, flow, pressure_level):
  mask = np.isclose(port_data[:,1].astype(float), float(pressure_level))
  subset = port_data[mask]
  if subset.size == 0:
    return np.full(port_data.shape[1]-2, np.nan, dtype=float)
  flows = subset[:,0].astype(float)
  vals  = subset[:,2:].astype(float)
  out = np.empty(vals.shape[1], dtype=float)
  for c in range(vals.shape[1]):
    out[c] = _interp1_extrap(flow, flows, vals[:,c])
  return out

def _interp(port_data, flow, pressure):
  flow = float(min(float(flow), MAX_FLOW))
  pressure = float(min(float(pressure), MAX_PRESSURE))

  pressures = np.unique(port_data[:,1].astype(float))
  pressures.sort()

  # 1) Interp/ekstrapolér i flow for hvert tryk-trin
  vals_by_p = np.vstack([_interp_flow_at_pressure(port_data, flow, p) for p in pressures])

  # 2) Interp/ekstrapolér i tryk for hver kolonne (oktavbånd + Total)
  out = np.empty(vals_by_p.shape[1], dtype=float)
  for c in range(vals_by_p.shape[1]):
    out[c] = _interp1_extrap(pressure, pressures, vals_by_p[:,c])
  return out

def interpolate_sound(flow, pressure):
  lw_s = _interp(lw_supply,   flow, pressure)
  lw_f = _interp(lw_extract,  flow, pressure)
  lw_u = _interp(lw_outside,  flow, pressure)
  lw_a = _interp(lw_exhaust,  flow, pressure)
  lp_c = _interp(lp_cabinet,  flow, pressure)
  return lw_s, lw_f, lw_u, lw_a, lp_c


@anvil.server.callable
def run_ECO190XL(flow, temp_out, pressure):
  sfp_j_m3 = interpolate_sfp(flow, pressure)
  sfp_w_m3 = sfp_j_m3 / 3600.0
  total_fan_w = sfp_j_m3 * (flow / 3600.0)
  eff, aft, sup = interpolate_heatrecovery(flow, temp_out)
  heat_power = calculate_power(flow, temp_out, sup)
  lw_s, lw_f, lw_u, lw_a, lp_c = interpolate_sound(flow, pressure)

  def _fmt(arr):
    try:
      return round(float(arr[-1]),1)
    except:
      return None
  result = {
      'SFP J/m³': round(sfp_j_m3,1),
      'SFP W/m³': round(sfp_w_m3,3),
      'Total fan W': round(total_fan_w,1),
      'Heat recovery %': round(eff,1),
      'Afkast temp °C': round(aft,1),
      'Supply temp °C': round(sup,1),
      'Heat power kW': round(heat_power,2),
      'Lw Total Supply dB(A)': _fmt(lw_s),
      'Lw Total Extract dB(A)': _fmt(lw_f),
      'Lw Total Outside dB(A)': _fmt(lw_u),
      'Lw Total Exhaust dB(A)': _fmt(lw_a),
      'Lp Total Cabinet dB(A)': _fmt(lp_c),
  }
  return result
