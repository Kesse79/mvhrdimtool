import anvil.server
import numpy as np
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import griddata

# ======== SFP-beregning ========
def p_sfp800(x):  return -0.0163 * x**2 + 3.1498 * x - 80.77
def p_sfp1000(x): return -0.0166 * x**2 + 3.4418 * x - 59.328
def p_sfp1200(x): return -0.0119 * x**2 + 2.5572 * x + 8.8081
def p_sfp1500(x): return -0.0147 * x**2 + 3.8282 * x - 26.403
def p_sfp1620(x): return -0.0117 * x**2 + 3.0481 * x + 40.934

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
  [100, 0, 91.6, 4.8, 18.3], [200, 0, 87.9, 5.4, 17.6], [300, 0, 85.7, 5.8, 17.1], [400, 0, 84.3, 6.0, 16.9],
  [100, 5, 91.4, 6.3, 18.7], [200, 5, 84.5, 6.9, 18.2], [300, 5, 85.6, 7.2, 17.8], [400, 5, 84.1, 7.4, 17.6],
  [100, 10, 91.3, 10.9, 19.1], [200, 10, 87.6, 11.2, 18.8], [300, 10, 85.5, 11.4, 18.5], [400, 10, 84.0, 11.6, 18.4],
  [100, 15, 91.1, 15.4, 19.6], [200, 15, 87.4, 15.6, 19.4], [300, 15, 85.3, 15.7, 19.3], [400, 15, 83.8, 15.8, 19.2],
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

# ======== LYD-DATA FRA DATABLADET ========
# Hver række: [flow, pressure, L63, L125, L250, L500, L1000, L2000, L4000, L8000, Total]

lw_supply = np.array([
  [ 70,  50, 27.05, 37.95, 50.13, 46.19, 45.92, 38.19, 28.65, 14.94, 52.96],
  [ 70,  70, 28.26, 38.32, 54.91, 47.66, 47.09, 39.65, 31.05, 14.13, 56.40],
  [100,  50, 31.59, 41.95, 52.42, 51.27, 52.18, 45.88, 39.89, 21.58, 57.32],
  [100,  70, 32.00, 42.15, 54.65, 51.98, 52.59, 46.40, 40.62, 19.89, 58.48],
  [126,  70, 43.67, 45.41, 62.16, 69.12, 57.58, 51.78, 47.78, 24.40, 70.28],
  [126, 100, 38.75, 46.04, 53.58, 62.14, 58.28, 52.43, 48.39, 25.04, 64.52],
  [162,  70, 40.16, 48.58, 55.49, 69.82, 61.68, 56.80, 53.63, 30.92, 70.87],
  [162, 100, 45.67, 49.35, 56.09, 69.83, 62.13, 57.36, 54.17, 32.20, 71.00],
  [216,  70, 46.92, 54.22, 60.15, 73.31, 67.35, 63.59, 61.39, 41.79, 75.04],
  [216, 100, 45.20, 53.86, 59.80, 73.73, 67.01, 63.22, 60.91, 41.06, 75.21],
])

lw_extract = np.array([
  [ 70,  50, 22.95, 24.70, 30.62, 30.67, 16.65,  7.74,  3.30,  2.77, 34.58],
  [ 70,  70, 23.84, 25.52, 35.37, 31.66, 18.58,  9.46,  4.25,  4.84, 37.48],
  [100,  50, 26.16, 28.08, 40.03, 35.74, 22.54, 12.65,  8.20,  8.37, 41.78],
  [100,  70, 27.69, 29.48, 40.08, 36.54, 21.49, 13.12,  9.14,  8.97, 42.15],
  [126,  70, 31.40, 32.88, 42.28, 41.36, 25.27, 16.73, 12.02, 11.72, 45.35],
  [126, 100, 41.13, 34.44, 43.07, 49.43, 26.66, 17.38, 14.31, 11.18, 50.94],
  [162,  70, 34.19, 34.15, 39.07, 47.68, 28.61, 19.65, 16.08, 12.58, 48.61],
  [162, 100, 33.20, 35.11, 40.22, 51.67, 30.11, 19.96, 17.17, 13.06, 52.15],
  [216,  70, 35.82, 37.47, 43.15, 50.56, 35.48, 26.84, 18.86, 15.02, 51.70],
  [216, 100, 36.21, 37.85, 42.96, 51.90, 34.01, 27.47, 19.96, 15.78, 52.75],
])

lw_outside = np.array([
  [ 70,  50, 20.63, 19.97, 23.68, 29.91, 17.98, 12.88,  1.31,  2.69, 31.81],
  [ 70,  70, 22.23, 19.71, 22.81, 30.41, 18.41,  7.71,  2.24,  3.37, 32.12],
  [100,  50, 24.74, 23.15, 27.12, 35.44, 19.86, 15.50,  5.61,  4.46, 36.69],
  [100,  70, 24.65, 22.72, 27.30, 35.06, 19.84, 16.12,  7.88,  5.62, 36.42],
  [126,  70, 40.95, 27.57, 35.01, 44.80, 24.49, 22.19, 12.22,  6.71, 46.70],
  [126, 100, 34.66, 26.95, 35.70, 43.90, 23.91, 21.95, 13.55, 12.71, 45.07],
  [162,  70, 34.26, 31.35, 33.98, 49.27, 32.05, 27.54, 14.27, 15.41, 49.69],
  [162, 100, 35.79, 30.19, 34.19, 48.73, 29.75, 28.24, 15.73, 17.80, 49.22],
  [216,  70, 38.45, 34.25, 39.90, 51.88, 36.14, 33.41, 23.61, 18.87, 52.55],
  [216, 100, 39.00, 35.29, 40.61, 53.11, 36.09, 33.60, 23.89, 21.87, 53.69],
])

lw_exhaust = np.array([
  [ 70,  50, 28.18, 38.11, 43.64, 44.93, 45.28, 38.29, 28.67, 18.69, 50.12],
  [ 70,  70, 29.58, 38.24, 46.96, 45.57, 46.27, 39.98, 31.31, 15.10, 51.67],
  [100,  50, 30.75, 40.45, 53.36, 49.95, 50.89, 45.81, 39.48, 14.82, 56.97],
  [100,  70, 33.23, 42.44, 52.27, 51.56, 51.87, 46.82, 40.92, 15.53, 57.37],
  [126,  70, 41.38, 44.59, 57.90, 61.12, 55.61, 51.37, 46.75, 22.31, 63.98],
  [126, 100, 37.36, 45.10, 52.91, 61.39, 58.03, 51.99, 47.49, 23.43, 63.91],
  [162,  70, 38.51, 46.82, 54.72, 73.02, 60.34, 56.76, 53.82, 31.26, 73.46],
  [162, 100, 41.05, 47.72, 55.11, 63.70, 60.59, 56.95, 53.74, 31.63, 66.65],
  [216,  70, 42.92, 55.36, 58.21, 70.08, 64.83, 62.00, 59.76, 39.94, 72.25],
  [216, 100, 42.07, 51.67, 58.81, 75.88, 65.50, 62.36, 60.11, 40.43, 76.62],
])

lp_cabinet = np.array([
  [ 70,  50, 14.41, 19.84, 26.53, 16.47, 17.70, 18.81, 20.24, 19.99, 29.90],
  [ 70,  70, 19.87, 23.99, 22.34, 17.88, 17.99, 19.08, 20.31, 20.02, 29.92],
  [100,  50, 17.21, 22.46, 28.09, 20.56, 18.80, 19.36, 20.31, 20.03, 31.44],
  [100,  70, 16.77, 24.53, 27.35, 21.97, 19.21, 19.42, 20.30, 19.99, 31.60],
  [126,  70, 17.19, 28.94, 26.05, 27.63, 21.48, 20.48, 20.44, 19.99, 33.65],
  [126, 100, 17.63, 32.14, 26.17, 27.68, 21.81, 20.76, 20.55, 20.03, 35.07],
  [162,  70, 27.41, 29.81, 28.26, 33.05, 26.96, 23.54, 21.30, 20.10, 37.16],
  [162, 100, 32.52, 29.39, 29.03, 30.65, 26.56, 23.79, 21.24, 20.04, 37.46],
  [216,  70, 25.06, 33.79, 32.74, 34.75, 29.57, 27.53, 23.02, 20.14, 39.72],
  [216, 100, 25.66, 33.49, 32.56, 36.36, 29.79, 27.56, 23.06, 20.11, 40.25],
])

lp_cabinet_light = np.array([
  [ 70,  50, 12.40, 16.51, 23.11, 25.15, 22.20, 18.40, 12.56, 10.66, 30.00],
  [ 70,  70, 12.50, 19.11, 28.41, 23.15, 22.30, 20.50, 13.56, 10.86, 32.10],
  [100,  50, 15.10, 22.81, 28.21, 29.95, 26.10, 23.90, 17.96, 10.46, 34.30],
  [100,  70, 15.80, 23.01, 28.81, 32.55, 27.80, 24.70, 18.36, 10.86, 34.60],
  [126,  70, 20.90, 26.61, 32.51, 39.45, 32.80, 29.80, 24.46, 11.86, 40.80],
  [126, 100, 18.10, 27.71, 29.21, 36.75, 30.20, 30.40, 25.16, 12.06, 39.90],
  [162,  70, 21.40, 32.71, 33.31, 47.75, 34.40, 35.30, 31.86, 15.86, 48.20],
  [162, 100, 22.10, 30.71, 34.01, 48.95, 36.50, 35.60, 22.26, 16.16, 49.30],
  [216,  70, 25.70, 36.81, 37.41, 47.25, 39.90, 40.90, 40.36, 24.86, 49.80],
  [216, 100, 26.80, 37.41, 38.21, 45.95, 40.20, 40.60, 40.56, 25.26, 49.20],
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
def run_ECO175PET(flow, temp_out, pressure):
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
