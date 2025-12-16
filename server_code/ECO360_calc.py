import anvil.server
import numpy as np
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import griddata

# ======== SFP-beregning (baseret på grafens tendenslinjer) ========
def p_sfp800(x):  return -0.0029 * x**2 + 0.931  * x + 32.039
def p_sfp1000(x): return -0.0032 * x**2 + 1.1085 * x + 68.357
def p_sfp1200(x): return -0.0035 * x**2 + 1.2791 * x + 88.694
def p_sfp1500(x): return -0.0037 * x**2 + 1.6113 * x + 108.56
def p_sfp1620(x): return -0.0041 * x**2 + 1.841  * x + 112.72

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
  for i in range(len(lst)-1):
    sfp_i, p_i = lst[i]; sfp_j, p_j = lst[i+1]
    if p_i <= pressure <= p_j:
      frac = (pressure - p_i) / (p_j - p_i)
      return sfp_i + frac * (sfp_j - sfp_i)
  # ekstrapolation
  if pressure < lst[0][1]:
    return ekstrapoler_under(pressure, lst[0], lst[1])
  else:
    return ekstrapoler_over(pressure, lst[-2], lst[-1])

# ======== Varmegenvinding ========
hr_data = np.array([
    [100,  0, 85.0,  3.1, 16.9], [200,  0, 84.0,  3.1, 16.9], [300,  0, 82.0,  3.6, 16.4], [400,  0, 79.0,  4.2, 15.8],
    [100,  5, 85.0,  7.3, 17.7], [200,  5, 84.0,  7.3, 17.7], [300,  5, 82.0,  7.7, 17.3], [400,  5, 79.0,  8.1, 16.9],
    [100, 10, 85.0, 11.6, 18.5], [200, 10, 85.0, 10.7, 18.3], [300, 10, 82.0, 11.8, 18.2], [400, 10, 79.0, 11.3, 17.7],
    [100, 15, 85.0, 14.9, 19.1], [200, 15, 84.0, 14.9, 19.1], [300, 15, 82.0, 15.9, 19.1], [400, 15, 79.0, 16.1, 19.0],
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
  mass_flow = (flow/3600) * rho_air
  return mass_flow * cp_air * (T_ude - T_supply)  # kW

# ======== LYD‑DATA ========
# Hver række: [flow, pressure, L63, L125, L250, L500, L1000, L2000, L4000, L8000, Total]

lw_supply = np.array([
    [126,  70, 40.44, 48.61, 58.34, 59.82, 56.87, 52.98, 43.76, 28.24, 63.62],
    [126, 100, 41.75, 49.88, 60.49, 61.63, 58.71, 54.23, 47.31, 30.72, 65.86],
    [162,  70, 40.46, 50.22, 58.63, 62.49, 57.90, 54.40, 45.79, 33.95, 65.82],
    [162, 100, 43.51, 52.23, 61.58, 63.87, 61.99, 58.05, 51.42, 34.88, 68.08],
    [216,  70, 44.51, 53.49, 66.38, 65.84, 64.06, 61.65, 54.91, 38.96, 71.32],
    [216, 100, 44.08, 54.97, 68.96, 66.58, 64.43, 62.39, 56.46, 43.13, 73.00],
    [200, 150, 46.20, 56.10, 65.20, 67.90, 66.30, 63.70, 57.50, 43.10, 72.90],
    [200, 200, 47.32, 58.19, 68.24, 68.90, 67.30, 66.10, 61.20, 48.20, 74.40],
    [250, 150, 46.70, 57.70, 69.20, 68.90, 67.50, 66.80, 62.40, 49.50, 74.80],
    [250, 200, 46.45, 58.43, 68.28, 70.90, 69.40, 68.70, 63.70, 51.50, 75.80],
])

lw_extract = np.array([
    [126,  70, 31.58, 41.22, 44.41, 47.14, 43.95, 37.23, 29.64, 23.51, 51.10],
    [126, 100, 33.54, 44.34, 53.08, 49.07, 46.19, 39.73, 32.69, 25.79, 56.12],
    [162,  70, 30.09, 43.72, 54.02, 48.74, 46.86, 40.84, 36.15, 29.58, 56.50],
    [162, 100, 37.78, 45.29, 55.27, 50.89, 49.30, 44.12, 41.92, 35.19, 58.12],
    [216,  70, 39.15, 47.34, 55.11, 51.82, 50.22, 44.83, 37.59, 30.11, 58.71],
    [216, 100, 38.57, 47.95, 53.58, 52.87, 51.68, 46.40, 41.20, 34.05, 58.71],
    [200, 150, 40.00, 49.50, 51.70, 54.40, 53.60, 47.70, 38.70, 29.90, 59.00],
    [200, 200, 39.24, 50.63, 52.82, 55.70, 54.60, 49.80, 41.80, 32.20, 60.60],
    [250, 150, 46.50, 52.30, 56.70, 57.40, 55.80, 50.40, 42.00, 32.90, 63.50],
    [250, 200, 42.52, 51.27, 54.60, 59.00, 57.30, 51.60, 43.20, 33.80, 63.10],
])

lw_outside = np.array([
    [126,  70, 29.76, 40.12, 47.88, 47.79, 46.41, 39.02, 25.89, 19.92, 52.85],
    [126, 100, 30.29, 41.90, 49.84, 49.22, 47.90, 41.45, 31.66, 24.76, 54.61],
    [162,  70, 31.39, 42.70, 54.17, 49.26, 48.94, 43.24, 39.53, 32.97, 57.30],
    [162, 100, 31.16, 43.42, 51.71, 50.87, 50.42, 44.03, 31.82, 22.52, 56.47],
    [216,  70, 41.81, 46.57, 58.70, 52.59, 52.60, 46.60, 36.48, 27.29, 61.15],
    [216, 100, 33.55, 44.31, 55.43, 54.30, 53.48, 48.05, 38.97, 30.24, 60.49],
    [200, 150, 34.90, 45.40, 55.20, 54.80, 55.40, 49.30, 41.60, 33.60, 60.60],
    [200, 200, 35.42, 46.00, 51.99, 56.00, 56.50, 51.20, 42.80, 34.20, 60.90],
    [250, 150, 40.42, 48.94, 56.22, 58.00, 58.40, 53.30, 44.70, 34.30, 62.20],
    [250, 200, 38.56, 47.23, 56.32, 58.70, 58.60, 53.50, 43.70, 32.20, 63.00],
])

lw_exhaust = np.array([
    [126,  70, 41.31, 47.87, 53.60, 60.67, 55.37, 51.37, 42.53, 27.41, 62.64],
    [126, 100, 44.69, 50.33, 58.54, 63.23, 57.93, 53.40, 45.39, 31.94, 66.20],
    [162,  70, 41.81, 50.55, 62.36, 62.46, 59.92, 56.96, 49.25, 33.88, 67.42],
    [162, 100, 43.51, 51.39, 59.49, 63.82, 59.94, 56.57, 48.75, 33.85, 67.15],
    [216,  70, 43.59, 54.20, 60.64, 65.21, 61.46, 59.32, 52.14, 39.74, 68.94],
    [216, 100, 43.71, 54.32, 63.16, 66.54, 62.76, 60.18, 54.12, 40.91, 70.24],
    [200, 150, 45.80, 55.90, 62.10, 68.20, 64.50, 61.70, 56.90, 45.30, 71.40],
    [200, 200, 47.13, 57.69, 63.88, 70.30, 66.40, 63.60, 59.10, 47.70, 73.40],
    [250, 150, 45.50, 56.70, 66.00, 70.30, 66.70, 64.80, 60.60, 49.50, 73.80],
    [250, 200, 47.00, 57.53, 66.06, 72.57, 68.87, 65.90, 61.87, 50.74, 75.50],
])

lp_cabinet = np.array([
    [126,  70, 22.51, 27.38, 32.69, 30.80, 30.77, 27.10, 22.87, 20.07, 36.75],
    [126, 100, 24.45, 28.60, 38.98, 32.30, 31.39, 27.11, 23.12, 20.16, 37.59],
    [162,  70, 23.44, 29.25, 40.61, 32.82, 32.23, 28.40, 23.78, 20.23, 38.22],
    [162, 100, 27.17, 30.87, 34.93, 33.95, 33.70, 29.71, 24.68, 20.34, 39.29],
    [216,  70, 24.72, 30.00, 38.97, 35.23, 35.18, 30.85, 25.37, 20.72, 41.40],
    [216, 100, 25.44, 30.75, 39.89, 36.04, 36.70, 31.89, 26.19, 20.92, 42.52],
    [200, 150, 26.10, 32.40, 37.80, 37.30, 37.50, 33.20, 27.10, 21.30, 43.00],
    [200, 200, 26.60, 33.70, 36.90, 39.10, 39.20, 35.10, 29.60, 22.10, 44.20],  # <-- her var den forkerte total, nu 44.20
    [250, 150, 25.90, 32.50, 38.60, 39.70, 39.70, 35.20, 29.20, 22.30, 44.70],
    [250, 200, 26.20, 34.20, 39.00, 41.40, 41.30, 37.10, 31.20, 23.40, 46.20],
])

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
  lw_s = _interp(lw_supply,  flow, pressure)
  lw_f = _interp(lw_extract, flow, pressure)
  lw_u = _interp(lw_outside, flow, pressure)
  lw_a = _interp(lw_exhaust, flow, pressure)
  lp_c = _interp(lp_cabinet, flow, pressure)
  return lw_s, lw_f, lw_u, lw_a, lp_c


@anvil.server.callable
def run_ECO360(flow, temp_out, pressure):
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
