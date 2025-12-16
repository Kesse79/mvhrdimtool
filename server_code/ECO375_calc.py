import anvil.server
import numpy as np
from scipy.interpolate import griddata
import numpy as np
from scipy.interpolate import griddata

import numpy as np
from scipy.interpolate import griddata

# ======== SFP-beregning ========
def p_sfp800(x): return -0.0033 * x**2 + 1.1627 * x + 30.145
def p_sfp1000(x): return -0.0041 * x**2 + 1.5492 * x + 38.23
def p_sfp1200(x): return -0.0037 * x**2 + 1.5755 * x + 69.213
def p_sfp1500(x): return -0.0042 * x**2 + 2.0067 * x + 78.332
def p_sfp1620(x): return -0.0048 * x**2 + 2.2927 * x + 73.647

POLY_LIST = [
    (800, p_sfp800),
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

# ======== LYD-DATA ========
# Hver række: [flow, pressure, L63, L125, L250, L500, L1000, L2000, L4000, L8000, Total]
lw_supply = np.array([
    [126,  70, 40.3, 52.0, 58.9, 61.0, 57.9, 53.9, 44.4, 28.8, 64.9],
    [126, 100, 42.9, 51.6, 61.0, 63.6, 60.9, 57.3, 48.4, 33.0, 67.5],
    [162,  70, 41.1, 51.3, 61.4, 64.5, 61.8, 57.3, 48.8, 32.7, 67.6],
    [162, 100, 43.1, 53.0, 61.9, 66.9, 64.4, 60.4, 52.3, 36.6, 69.8],
    [216,  70, 42.1, 54.3, 66.8, 68.5, 67.4, 63.2, 55.2, 40.5, 72.6],
    [216, 100, 44.5, 56.2, 64.8, 70.7, 68.7, 64.5, 57.3, 42.1, 74.2],
    [250, 150, 46.3, 58.4, 70.7, 73.5, 72.3, 69.2, 62.0, 47.6, 77.5],
    [250, 200, 48.4, 60.0, 70.8, 75.1, 73.9, 71.1, 63.9, 50.0, 78.7],
])

lw_extract = np.array([
    [126,  70, 24.2, 30.6, 37.1, 37.5, 34.3, 29.5, 22.2, 20.3, 41.9],
    [126, 100, 25.8, 32.3, 40.2, 40.4, 37.4, 32.5, 23.3, 20.0, 44.9],
    [162,  70, 26.4, 31.2, 39.0, 39.9, 36.6, 32.4, 23.4, 20.4, 43.8],
    [162, 100, 25.8, 33.0, 41.9, 42.6, 40.0, 35.2, 24.7, 21.3, 46.7],
    [216,  70, 25.5, 34.1, 45.8, 43.6, 41.7, 37.5, 26.5, 21.3, 48.9],
    [216, 100, 26.5, 35.9, 48.1, 46.5, 43.8, 39.6, 28.5, 21.4, 51.3],
    [250, 150, 29.5, 39.0, 49.9, 50.5, 49.1, 45.7, 34.8, 23.5, 55.1],
    [250, 200, 32.0, 40.7, 51.8, 52.0, 51.0, 48.2, 38.4, 25.8, 56.8],
])

lw_outside = np.array([
    [126,  70, 24.2, 30.5, 38.7, 41.2, 36.1, 28.7, 23.0, 20.4, 44.3],
    [126, 100, 27.4, 31.8, 39.8, 43.5, 39.1, 31.2, 23.0, 20.0, 46.4],
    [162,  70, 26.7, 31.3, 38.7, 43.3, 38.8, 30.9, 23.3, 19.6, 46.1],
    [162, 100, 28.2, 33.1, 41.2, 46.3, 41.9, 33.9, 24.3, 19.9, 49.0],
    [216,  70, 36.1, 37.6, 43.2, 46.1, 43.3, 35.6, 26.2, 21.3, 49.4],
    [216, 100, 32.6, 37.2, 43.5, 47.3, 44.9, 37.2, 27.0, 20.6, 50.4],
    [250, 150, 42.4, 47.3, 51.8, 51.9, 50.4, 44.2, 34.2, 25.9, 56.5],
    [250, 200, 43.9, 48.9, 56.9, 53.9, 52.4, 47.5, 37.5, 25.9, 59.7],
])

lw_exhaust = np.array([
    [126,  70, 39.3, 49.6, 57.9, 60.7, 56.9, 51.5, 43.0, 27.2, 63.7],
    [126, 100, 42.3, 52.1, 60.8, 63.2, 59.7, 54.9, 47.1, 31.5, 66.4],
    [162,  70, 38.3, 52.5, 62.4, 63.2, 61.0, 55.5, 46.5, 30.3, 67.2],
    [162, 100, 42.2, 51.7, 62.9, 65.2, 63.3, 58.7, 50.1, 35.0, 68.9],
    [216,  70, 38.9, 54.2, 65.9, 68.2, 65.9, 61.7, 53.4, 39.1, 71.8],
    [216, 100, 42.3, 55.6, 67.8, 69.5, 67.7, 63.6, 55.8, 41.9, 73.5],
    [250, 150, 45.0, 58.0, 71.2, 73.4, 71.8, 68.5, 61.5, 48.0, 77.6],
    [250, 200, 48.2, 59.9, 71.5, 75.0, 73.9, 70.8, 64.0, 50.3, 78.7],
])

lp_cabinet = np.array([
    [126,  70, 25.0, 32.9, 37.4, 35.1, 35.4, 29.6, 23.3, 21.9, 34.4],
    [126, 100, 26.5, 33.9, 39.8, 38.6, 37.5, 32.6, 27.3, 23.8, 36.0],
    [162,  70, 26.6, 33.7, 40.5, 38.6, 37.7, 32.8, 27.4, 24.1, 36.0],
    [162, 100, 27.2, 34.7, 41.6, 39.5, 38.6, 33.5, 25.6, 23.8, 37.5],
    [216,  70, 24.9, 36.0, 45.9, 41.7, 41.1, 35.9, 27.0, 22.4, 39.5],
    [216, 100, 25.9, 36.6, 47.1, 43.1, 42.4, 37.1, 28.2, 23.0, 40.7],
    [250, 150, 29.2, 39.8, 50.9, 47.0, 45.9, 41.6, 32.4, 24.3, 45.2],
    [250, 200, 32.9, 41.1, 51.0, 48.1, 47.7, 43.4, 34.5, 25.9, 45.8],
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
def run_ECO375(flow, temp_out, pressure):
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
