from __future__ import annotations

import math


UTCI_NO_HEAT_STRESS_C = 26.0


def apparent_temperature_c(temp_c: float, humidity_pct: float, wind_mps: float) -> float:
    """Australian BOM apparent-temperature approximation (legacy display helper)."""
    humidity = min(100.0, max(0.0, humidity_pct))
    wind = max(0.0, wind_mps)
    vapor_pressure = humidity / 100.0 * 6.105 * math.exp(17.27 * temp_c / (237.7 + temp_c))
    return temp_c + 0.33 * vapor_pressure - 0.70 * wind - 4.0


def mean_radiant_temperature_c(
    *, air_temperature_c: float, direct_normal_radiation_wm2: float,
    diffuse_radiation_wm2: float = 0.0,
    solar_elevation_deg: float, solar_horizontal_angle_deg: float,
    shade_fraction: float, sky_view_factor: float = 1.0,
    relative_humidity_pct: float = 50.0, cloud_cover_pct: float = 0.0,
    surface_albedo: float = 0.20,
) -> float:
    """Estimate MRT using the ASHRAE 55 SolarCal effective-radiant-field model.

    The long-wave baseline is set equal to air temperature and LiDAR shade is
    applied as short-wave transmittance. This remains an engineering
    approximation, rather than a full urban radiation model such as SOLWEIG.
    """
    altitude = min(90.0, max(0.0, solar_elevation_deg))
    direct = min(1000.0, max(0.0, direct_normal_radiation_wm2))
    transmittance = 1.0 - min(1.0, max(0.0, shade_fraction))
    svf = min(1.0, max(0.0, sky_view_factor))
    # Brutsaert clear-sky emissivity with a standard cloud correction supplies
    # the long-wave sky term; surrounding surfaces are assumed at air temperature.
    sigma = 5.670374419e-8
    air_k = air_temperature_c + 273.15
    vapor_hpa = max(0.01, 10.0 * relative_humidity_pct / 100.0 * 0.6108 * math.exp(17.27 * air_temperature_c / (air_temperature_c + 237.3)))
    clear_emissivity = 1.24 * (vapor_hpa / air_k) ** (1.0 / 7.0)
    cloud = min(1.0, max(0.0, cloud_cover_pct / 100.0))
    sky_emissivity = min(1.0, clear_emissivity * (1.0 + 0.22 * cloud * cloud))
    sky_longwave = sky_emissivity * sigma * air_k**4
    surface_longwave = 0.95 * sigma * air_k**4
    baseline_k = ((svf * sky_longwave + (1.0 - svf) * surface_longwave) / (0.95 * sigma)) ** 0.25
    baseline_c = baseline_k - 273.15
    if altitude <= 0.0 or direct < 1.0 or transmittance <= 0.0:
        return float(baseline_c)
    # Import the scientific model only when a calculation is requested. This
    # keeps the public interface responsive while users edit route controls.
    from pythermalcomfort.models import solar_gain
    result = solar_gain(
        sol_altitude=altitude,
        sharp=min(180.0, max(0.0, solar_horizontal_angle_deg)),
        sol_radiation_dir=direct,
        sol_transmittance=transmittance,
        f_svv=svf,
        f_bes=1.0,
        posture="standing",
        floor_reflectance=min(1.0, max(0.0, surface_albedo)),
        round_output=False,
    )
    # SolarCal assumes diffuse irradiance = 0.2 × DNI. Correct its ERF using
    # the measured Open-Meteo diffuse horizontal irradiance.
    diffuse_delta = max(0.0, diffuse_radiation_wm2) - 0.2 * direct
    f_eff, shortwave_absorptivity, longwave_absorptivity, hr = 0.725, 0.7, 0.95, 6.0
    diffuse_erf_correction = (
        f_eff * svf * 0.5 * transmittance * diffuse_delta * (1.0 + surface_albedo)
        * shortwave_absorptivity / longwave_absorptivity
    )
    diffuse_delta_mrt = diffuse_erf_correction / (hr * f_eff)
    return float(baseline_c + result.delta_mrt + diffuse_delta_mrt)


def utci_c(*, temp_c: float, mean_radiant_temp_c: float, humidity_pct: float, wind_10m_mps: float) -> float:
    """Calculate UTCI using its standard 10 m wind convention."""
    from pythermalcomfort.models import utci
    result = utci(
        tdb=temp_c,
        tr=mean_radiant_temp_c,
        v=min(17.0, max(0.5, wind_10m_mps)),
        rh=min(100.0, max(0.0, humidity_pct)),
        limit_inputs=True,
        round_output=False,
    )
    return float(result.utci)


def utci_stress_category(value_c: float) -> str:
    if value_c <= 26.0:
        return "No heat stress"
    if value_c <= 32.0:
        return "Moderate heat stress"
    if value_c <= 38.0:
        return "Strong heat stress"
    if value_c <= 46.0:
        return "Very strong heat stress"
    return "Extreme heat stress"


def segment_heat_cost(
    *, temp_c: float, humidity_pct: float, wind_mps: float,
    radiation_wm2: float = 0.0,
    direct_normal_radiation_wm2: float | None = None,
    diffuse_radiation_wm2: float = 0.0,
    solar_elevation_deg: float = 45.0,
    solar_horizontal_angle_deg: float = 90.0,
    shade_fraction: float, sky_view_factor: float = 1.0,
    cloud_cover_pct: float = 0.0, surface_albedo: float = 0.20,
    duration_minutes: float, profile=None,
) -> float:
    """Return UTCI heat-exposure load in degree-minutes above 26 °C."""
    direct = radiation_wm2 if direct_normal_radiation_wm2 is None else direct_normal_radiation_wm2
    mrt = mean_radiant_temperature_c(
        air_temperature_c=temp_c,
        direct_normal_radiation_wm2=direct,
        diffuse_radiation_wm2=diffuse_radiation_wm2,
        solar_elevation_deg=solar_elevation_deg,
        solar_horizontal_angle_deg=solar_horizontal_angle_deg,
        shade_fraction=shade_fraction,
        sky_view_factor=sky_view_factor,
        relative_humidity_pct=humidity_pct,
        cloud_cover_pct=cloud_cover_pct,
        surface_albedo=surface_albedo,
    )
    value = utci_c(
        temp_c=temp_c, mean_radiant_temp_c=mrt,
        humidity_pct=humidity_pct, wind_10m_mps=wind_mps,
    )
    return max(0.0, value - UTCI_NO_HEAT_STRESS_C) * max(0.0, duration_minutes)


def risk_label(utci_value_c: float) -> str:
    return utci_stress_category(utci_value_c)
