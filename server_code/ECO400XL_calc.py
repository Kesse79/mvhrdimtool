import anvil.server
import numpy as np
from scipy.interpolate import griddata

@anvil.server.callable
def run_ECO300XL(flow, temp, pressure):
  def p_sfp800(x): return -0.0018 * x**2 + 0.7509 * x + 33.072
  def p_sfp1000(x): return -0.0023 * x**2 + 1.1327 * x + 13.397
  def p_sfp1500(x): return -0.0019 * x**2 + 1.0792 * x + 97.638
  def p_sfp1620(x): return -0.0018 * x**2 + 1.1262 * x + 97.84
  def p_sfp1800(x): return -0.0019 * x**2 + 1.2243 * x + 121.21
  def p_sfp2100(x): return -0.0016 * x**2 + 1.173 * x + 166.27

  POLY_LIST = [
      (800, p_sfp800),
      (1000, p_sfp1000),
      (1500, p_sfp1500),
      (1620, p_sfp1620),
      (1800, p_sfp1800),
      (2100, p_sfp2100),
  ]

  hr_data = np.array([
      [100,  0, 91.8,  1.6, 18.4], [200,  0, 87.8,  2.4, 17.6], [300,  0, 84.3,  3.1, 16.9],
      [400,  0, 81.3,  3.7, 16.3], [500,  0, 79.0,  4.2, 15.8],
      [100,  5, 91.9,  6.2, 18.8], [200,  5, 88.1,  6.8, 18.2], [300,  5, 84.8,  7.3, 17.7],
      [400,  5, 82.2,  7.7, 17.3], [500,  5, 80.3,  8.0, 17.0],
      [100, 10, 92.0, 10.8, 19.2], [200, 10, 88.3, 11.2, 18.8], [300, 10, 85.3, 11.5, 18.5],
      [400, 10, 83.1, 11.7, 18.3], [500, 10, 81.6, 11.8, 18.2],
      [100, 15, 92.1, 15.4, 19.6], [200, 15, 88.6, 15.6, 19.4], [300, 15, 85.9, 15.7, 19.3],
      [400, 15, 83.9, 15.8, 19.2], [500, 15, 82.8, 15.9, 19.1],
  ])
  points_hr = hr_data[:, :2]
  eff_hr = hr_data[:, 2]
  T_afkast_hr = hr_data[:, 3]
  T_supply_hr = hr_data[:, 4]
  cp_air = 1.005
  rho_air = 1.2

  # Dummy sound data
# ======== LYD-DATA ========
  lw_supply = np.array([
      [126,  70, 32.7, 43.6, 45.3, 48.1, 48.2, 46.2, 39.0, 21.4, 52.0],
      [126, 100, 35.1, 43.8, 46.9, 47.7, 49.2, 47.7, 38.6, 23.2, 53.2],
      [162,  70, 30.4, 39.6, 46.0, 47.2, 48.8, 47.3, 37.0, 22.6, 54.4],
      [162, 100, 34.7, 43.6, 49.7, 48.9, 49.4, 49.4, 40.4, 25.3, 55.3],
      [200, 150, 29.7, 41.8, 49.4, 47.7, 47.9, 47.1, 37.3, 21.4, 54.6],
      [200, 200, 30.8, 43.5, 51.6, 50.3, 49.2, 49.8, 42.0, 25.6, 56.5],
      [216,  70, 32.3, 42.1, 42.8, 48.8, 49.2, 41.3, 28.9, 26.8, 57.0],
      [216, 100, 35.0, 42.6, 58.0, 52.3, 53.2, 52.6, 44.2, 29.3, 61.5],
      [250, 150, 31.0, 42.3, 51.8, 51.0, 51.1, 51.3, 41.2, 25.3, 58.0],
      [250, 200, 30.2, 42.6, 56.3, 50.4, 49.9, 52.5, 43.5, 27.8, 60.7],
      [300, 150, 32.0, 42.4, 52.9, 53.7, 52.8, 54.4, 48.0, 31.1, 58.5],
      [300, 200, 33.6, 43.0, 54.3, 57.4, 53.6, 54.4, 47.6, 31.4, 62.0],
      [400, 200, 37.5, 43.9, 54.6, 60.8, 57.1, 59.3, 53.5, 37.5, 66.6],
      [458, 100, 43.2, 50.2, 55.9, 64.2, 63.4, 66.0, 63.8, 49.7, 70.0],
      [600, 250, 45.2, 52.4, 59.1, 69.6, 65.4, 66.9, 62.8, 50.6, 73.5],
  ])

  lw_extract = np.array([
      [126,  70, 25.1, 34.2, 43.2, 36.0, 30.1, 24.1, 14.0, 11.5, 43.6],
      [126, 100, 22.6, 33.4, 40.6, 35.9, 31.2, 25.6, 18.5, 12.4, 43.4],
      [162,  70, 21.3, 32.2, 41.5, 34.8, 32.0, 24.6, 14.0, 10.9, 41.1],
      [162, 100, 22.7, 34.3, 43.3, 37.1, 33.6, 27.5, 15.6, 11.5, 44.2],
      [200, 150, 22.7, 32.7, 44.0, 33.8, 28.6, 26.1, 19.5, 13.3, 46.4],
      [200, 200, 25.3, 34.1, 44.0, 38.3, 36.1, 31.5, 25.6, 16.9, 44.9],
      [216,  70, 22.1, 29.3, 41.3, 35.6, 32.3, 26.9, 15.0, 11.1, 43.3],
      [216, 100, 22.3, 31.2, 45.2, 38.3, 35.7, 31.0, 18.1, 11.1, 46.3],
      [250, 150, 23.8, 30.9, 42.4, 36.0, 35.5, 31.8, 25.9, 17.7, 45.5],
      [250, 200, 24.5, 32.8, 45.1, 37.2, 36.8, 32.1, 24.9, 15.0, 46.2],
      [300, 150, 29.3, 33.1, 45.6, 38.2, 35.7, 31.6, 22.6, 14.5, 47.6],
      [300, 200, 22.8, 32.8, 44.3, 40.5, 38.0, 35.3, 27.5, 15.7, 49.4],
      [400, 200, 33.5, 36.0, 46.8, 47.2, 42.5, 37.4, 28.8, 19.3, 52.6],
      [458, 100, 26.7, 36.1, 49.1, 54.0, 44.4, 41.1, 31.4, 17.5, 55.7],
      [600, 250, 31.8, 40.5, 51.2, 51.6, 48.2, 44.2, 36.7, 25.9, 56.3],
  ])

  lw_outside = np.array([
      [126,  70, 24.6, 33.7, 37.3, 33.8, 30.2, 22.3, 16.0, 18.5, 40.7],
      [126, 100, 24.3, 34.6, 43.9, 38.7, 34.1, 25.1, 15.1, 11.6, 45.3],
      [162,  70, 26.6, 33.4, 44.5, 36.7, 35.1, 26.6, 17.5, 11.0, 43.8],
      [162, 100, 25.7, 30.9, 40.2, 36.7, 33.2, 25.0, 16.0, 11.2, 43.8],
      [200, 150, 25.0, 32.8, 49.8, 38.0, 35.7, 28.4, 20.5, 12.7, 51.3],
      [200, 200, 27.3, 33.3, 44.5, 40.9, 40.3, 32.6, 24.4, 13.0, 44.5],
      [216,  70, 23.8, 29.2, 42.7, 37.6, 34.1, 26.3, 15.6, 11.1, 45.7],
      [216, 100, 28.3, 38.5, 47.9, 40.6, 39.3, 31.7, 19.3, 11.2, 48.5],
      [250, 150, 28.0, 32.9, 42.4, 38.7, 37.0, 30.5, 21.4, 12.3, 44.9],
      [250, 200, 27.3, 33.2, 43.0, 39.5, 38.2, 30.5, 22.4, 12.6, 46.9],
      [300, 150, 27.8, 33.3, 43.5, 39.4, 39.3, 32.6, 23.2, 13.0, 46.6],
      [300, 200, 28.1, 34.0, 46.7, 43.5, 40.2, 32.4, 24.2, 14.1, 49.9],
      [400, 200, 26.7, 35.9, 48.8, 51.5, 47.5, 37.6, 30.0, 16.8, 53.1],
      [458, 100, 28.7, 36.8, 47.2, 54.1, 50.3, 40.1, 30.6, 18.0, 57.3],
      [600, 250, 32.6, 41.1, 48.2, 58.6, 55.8, 46.6, 38.2, 28.1, 61.5],
  ])

  lw_exhaust = np.array([
      [126,  70, 30.5, 38.5, 45.8, 45.4, 46.0, 44.4, 35.2, 18.9, 50.7],
      [126, 100, 32.6, 38.1, 43.6, 44.6, 45.4, 44.0, 35.0, 18.3, 51.8],
      [162,  70, 30.2, 37.5, 45.7, 44.9, 46.6, 45.8, 36.0, 19.8, 51.8],
      [162, 100, 32.2, 38.8, 46.7, 47.2, 47.6, 48.1, 39.4, 21.8, 54.0],
      [200, 150, 33.4, 37.3, 46.3, 46.5, 47.3, 47.2, 37.2, 20.9, 52.0],
      [200, 200, 32.4, 39.4, 52.7, 48.2, 48.2, 48.7, 40.2, 23.3, 55.9],
      [216,  70, 30.2, 37.6, 49.3, 48.1, 48.8, 46.6, 32.9, 17.2, 55.7],
      [216, 100, 32.7, 39.9, 52.2, 49.1, 52.0, 51.3, 43.3, 27.8, 57.7],
      [250, 150, 31.2, 38.9, 56.1, 48.2, 50.1, 51.9, 40.3, 26.6, 56.8],
      [250, 200, 32.6, 39.1, 50.8, 49.7, 50.2, 51.8, 42.8, 26.6, 58.1],
      [300, 150, 31.4, 39.6, 50.1, 52.6, 51.9, 55.5, 47.5, 30.8, 59.2],
      [300, 200, 35.5, 41.4, 52.2, 54.2, 52.5, 55.9, 48.0, 31.8, 60.2],
      [400, 200, 38.1, 43.0, 53.7, 59.8, 60.0, 60.8, 56.7, 40.8, 65.3],
      [458, 100, 40.1, 44.8, 54.5, 60.7, 61.0, 63.3, 60.0, 45.0, 69.4],
      [600, 250, 46.0, 51.4, 58.3, 65.4, 66.1, 67.8, 64.4, 53.0, 72.7],
  ])

  lp_cabinet = np.array([
      [126,  70, 22.7, 29.8, 30.8, 25.5, 27.9, 22.5, 30.6, 12.0, 35.3],
      [126, 100, 20.8, 31.4, 33.5, 28.2, 28.7, 24.4, 16.5, 11.5, 37.2],
      [162,  70, 21.1, 28.1, 31.3, 26.1, 28.7, 22.8, 16.3, 11.9, 35.7],
      [162, 100, 21.9, 28.5, 32.0, 28.8, 29.6, 24.4, 17.1, 11.9, 37.8],
      [200, 150, 22.0, 27.7, 32.4, 27.9, 29.4, 25.0, 18.9, 12.8, 36.6],
      [200, 200, 23.9, 29.5, 36.8, 30.7, 32.0, 28.1, 22.4, 13.4, 38.7],
      [216,  70, 20.5, 26.7, 32.9, 27.9, 30.6, 25.9, 18.4, 17.8, 37.0],
      [216, 100, 20.7, 26.3, 36.0, 29.4, 31.9, 28.7, 20.5, 12.5, 38.5],
      [250, 150, 22.0, 29.1, 34.0, 29.5, 32.0, 28.3, 21.5, 12.5, 38.5],
      [250, 200, 25.2, 29.1, 36.2, 30.7, 33.3, 29.1, 23.6, 15.0, 39.3],
      [300, 150, 23.4, 29.4, 36.3, 32.2, 34.2, 31.7, 25.8, 15.8, 39.7],
      [300, 200, 27.1, 30.2, 38.3, 35.4, 35.4, 31.9, 25.7, 14.7, 42.9],
      [400, 200, 30.8, 33.7, 40.7, 43.7, 40.9, 36.7, 33.0, 21.6, 48.0],
      [458, 100, 27.8, 32.4, 38.3, 43.7, 42.7, 38.2, 35.5, 23.5, 48.3],
      [600, 250, 32.1, 37.9, 41.1, 47.7, 48.6, 44.4, 43.2, 34.1, 53.7],
  ])
  def interpolate_sfp(flow, pressure):
    lst = [(sfp, poly(flow)) for sfp, poly in POLY_LIST]
    lst.sort(key=lambda x: x[1])
    for i in range(len(lst) - 1):
      sfp_i, p_i = lst[i]
      sfp_j, p_j = lst[i + 1]
      if p_i <= pressure <= p_j:
        frac = (pressure - p_i) / (p_j - p_i)
        return sfp_i + frac * (sfp_j - sfp_i)
    # extrapolate
    if pressure < lst[0][1]:
      sfp_i, p_i = lst[0]
      sfp_j, p_j = lst[1]
    else:
      sfp_i, p_i = lst[-2]
      sfp_j, p_j = lst[-1]
    frac = (pressure - p_i) / (p_j - p_i)
    return sfp_i + frac * (sfp_j - sfp_i)

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
    eff = _interp_heat(eff_hr, flow, T_out)
    aft = _interp_heat(T_afkast_hr, flow, T_out)
    sup = _interp_heat(T_supply_hr, flow, T_out)
    return eff, aft, sup

  def calculate_power(flow, T_ude, T_supply):
    mass_flow = (flow / 3600) * rho_air
    return mass_flow * cp_air * (T_ude - T_supply)

  def _interp(port_data, flow, pressure):
    pt = np.array([[flow, pressure]])
    vals = griddata(port_data[:, :2], port_data[:, -1], pt, method='linear')
    return vals[0] if vals is not None else np.nan

  def interpolate_sound(flow, pressure):
    lw_s = _interp(lw_supply, flow, pressure)
    lw_f = _interp(lw_extract, flow, pressure)
    lw_u = _interp(lw_outside, flow, pressure)
    lw_a = _interp(lw_exhaust, flow, pressure)
    lp_c = _interp(lp_cabinet, flow, pressure)
    return lw_s, lw_f, lw_u, lw_a, lp_c

  def safe_fmt(val):
    return f"{val:.1f} dB(A)" if not np.isnan(val) else "n/a"

  # === MAIN LOGIC ===
  if not (50 <= flow <= 500):
    raise ValueError("Luftmængden skal være mellem 50 og 500 m³/h")
  if pressure < 0:
    raise ValueError("Tryk kan ikke være negativt")

  sfp_j_m3 = interpolate_sfp(flow, pressure)
  sfp_w_m3 = sfp_j_m3 / 3600.0
  total_fan_w = sfp_j_m3 * (flow / 3600.0)

  eff, aft, sup = interpolate_heatrecovery(flow, temp)
  heat_kw = calculate_power(flow, temp, sup)

  eff7 = _interp_heat(eff_hr, flow, 7)
  savings_kwh = 37.5 * flow * (eff7 / 100.0)

  lw_s, lw_f, lw_u, lw_a, lp_c = interpolate_sound(flow, pressure)

  return {
      "sfp i j/m3": round(sfp_j_m3, 1),
      "sfp i w/m3": round(sfp_w_m3, 3),
      "effekt forbrug ventilator i watt": round(total_fan_w, 1),
      "varmegenvinding i %": round(eff, 1),
      "Afkast temperatur °C": round(aft, 1),
      "Tilluftstemperatur °C": round(sup, 1),
      "Varme genvinding kW": round(heat_kw, 2),
      "Årlig energibesparelse i kwh ift. nat. vent": round(savings_kwh),
      "lw_tilluft": safe_fmt(lw_s),
      "lw_udsugning": safe_fmt(lw_f),
      "lw_friskluft": safe_fmt(lw_u),
      "lw_afkast": safe_fmt(lw_a),
      "lp_kabinet": safe_fmt(lp_c)
  }
