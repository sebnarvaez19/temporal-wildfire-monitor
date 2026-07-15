import os
import sys

# Workaround for GDAL, GEOS, and PROJ DLL loading issues in Windows Streamlit environments
if sys.platform == 'win32':
    import site
    sp_paths = []
    try:
        sp_paths.extend(site.getsitepackages())
    except AttributeError:
        pass
    # Local .venv fallback path relative to this script
    venv_sp = os.path.abspath(os.path.join(os.path.dirname(__file__), '.venv', 'Lib', 'site-packages'))
    sp_paths.append(venv_sp)
    
    for sp in sp_paths:
        if os.path.isdir(sp):
            for folder in os.listdir(sp):
                if folder.endswith('.libs'):
                    libs_path = os.path.join(sp, folder)
                    if os.path.isdir(libs_path):
                        try:
                            os.add_dll_directory(libs_path)
                        except Exception:
                            pass

import streamlit as st
import folium
from streamlit_folium import st_folium
import geopandas as gpd
import pandas as pd
import numpy as np
import plotly.graph_objects as go


# Set page configuration
st.set_page_config(
    page_title="Monitor de Incendios Forestales - Atlántico",
    page_icon="public/logo.png",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for rich aesthetics and modern typography
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap');
    
    /* Styling headers */
    h1, h2, h3 {
        font-family: 'Outfit', 'Inter', sans-serif;
        color: #1e293b;
        font-weight: 700;
    }
    
    p, span, div, label {
        font-family: 'Inter', sans-serif;
    }
    
    /* Card Container */
    .metric-card {
        background: #ffffff;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border-left: 5px solid #3b82f6; /* Default Blue */
        margin-bottom: 15px;
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
    }
    .card-title {
        font-size: 12px;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
        margin-bottom: 4px;
    }
    .card-value {
        font-size: 24px;
        font-weight: 700;
        color: #0f172a;
    }
    .card-subtitle {
        font-size: 11px;
        color: #94a3b8;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Title Section
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.image("public/logo.png", width=110)
with col_title:
    st.markdown("""
        <div style="padding-top: 10px;">
            <h1 style="margin:0; font-family:'Outfit', sans-serif; font-size:32px; line-height: 1.1;">Monitor de Incendios Forestales - Atlántico</h1>
            <p style="margin:5px 0 0 0; font-size:16px; color:#475569; line-height: 1.3;">Visualización interactiva y análisis temporal de incendios forestales monitoreados en el Departamento del Atlántico, Colombia.</p>
        </div>
    """, unsafe_allow_html=True)
st.write("") # Spacer

# Helper to sanitize DataFrames/GeoDataFrames for JSON serialization
def sanitize_for_serialization(df):
    df_clean = df.copy()
    for col in df_clean.columns:
        if col != 'geometry':
            # Convert list/ndarray columns to string
            if df_clean[col].dtype == 'object':
                df_clean[col] = df_clean[col].apply(
                    lambda x: ', '.join(x) if isinstance(x, (list, np.ndarray)) else str(x)
                )
            # Convert datetime columns to string
            elif pd.api.types.is_datetime64_any_dtype(df_clean[col]):
                df_clean[col] = df_clean[col].astype(str)
    return df_clean

# Load and preprocess datasets
@st.cache_data
def load_and_preprocess_data():
    # Load municipalities (utf-8 encoding ensures Spanish characters decode correctly)
    gdf_m = gpd.read_file('data/municipalities.geojson', encoding='utf-8')
    gdf_m['MpNombre'] = gdf_m['MpNombre'].str.strip()
    
    # Load wildfires
    gdf_w = gpd.read_file('data/wildfire_2026-01-01_2026-07-14.geojson')
    
    # Parse acquisition date and add columns to gdf_w
    acq_dates = pd.to_datetime(gdf_w['oldest_acquisition'], errors='coerce')
    gdf_w['acq_date'] = acq_dates
    gdf_w['year_month'] = acq_dates.dt.strftime('%Y-%m')
    
    # Map months to Spanish names
    SPANISH_MONTHS = {
        1: 'Ene', 2: 'Feb', 3: 'Mar', 4: 'Abr',
        5: 'May', 6: 'Jun', 7: 'Jul', 8: 'Ago',
        9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dic'
    }
    gdf_w['month_name'] = acq_dates.dt.month.map(SPANISH_MONTHS) + ' ' + acq_dates.dt.year.astype(str)
    
    # Optimize rendering performance: simplify wildfire geometries
    # A tolerance of 0.0002 (~20m) preserves shape details while dropping ~75% of coordinate count
    gdf_w_geom_simplified = gdf_w.copy()
    gdf_w_geom_simplified['geometry'] = gdf_w.geometry.simplify(0.0002, preserve_topology=True)
    
    # Reproject municipalities to EPSG:9377 (Colombia Origen Nacional) to compute metric area/perimeter
    gdf_m_proj = gdf_m.to_crs(epsg=9377)
    gdf_m['area_m2'] = gdf_m_proj.geometry.area
    gdf_m['perimeter_m'] = gdf_m_proj.geometry.length
    
    # Spatial join to determine which municipality each wildfire intersects
    # We do the join using the original EPSG:4326 geometries for accuracy
    joined = gpd.sjoin(gdf_w, gdf_m[['MpNombre', 'geometry']], how='inner', predicate='intersects')
    
    # Update joined dataset with parsed date columns
    joined['acq_date'] = pd.to_datetime(joined['oldest_acquisition'], errors='coerce')
    joined['year_month'] = joined['acq_date'].dt.strftime('%Y-%m')
    joined['month_name'] = joined['acq_date'].dt.month.map(SPANISH_MONTHS) + ' ' + joined['acq_date'].dt.year.astype(str)
    
    # Apply JSON serialization sanitization to prevent Folium serialization errors
    gdf_w_geom_simplified = sanitize_for_serialization(gdf_w_geom_simplified)
    joined = sanitize_for_serialization(joined)
    
    # Calculate Departamento del Atlántico overall union stats
    union_geom = gdf_m_proj.geometry.union_all() if hasattr(gdf_m_proj.geometry, 'union_all') else gdf_m_proj.geometry.unary_union
    dept_area = union_geom.area
    dept_perimeter = union_geom.length
    
    return gdf_m, gdf_w_geom_simplified, joined, dept_area, dept_perimeter

try:
    gdf_m, gdf_w, joined, dept_area, dept_perimeter = load_and_preprocess_data()
except Exception as e:
    st.error(f"Error cargando los datos del sistema: {e}")
    st.stop()

# Initialize session state for selected municipality
if "selected_municipality" not in st.session_state:
    st.session_state["selected_municipality"] = "Todos"

# Sidebar Configuration
st.sidebar.header("⚙️ Configuración")
st.sidebar.write("### Opciones del Mapa")
map_style = st.sidebar.radio(
    "Capa base del mapa:",
    options=["Esri World Imagery", "CartoDB Positron", "CartoDB Dark Matter", "Jawg Lagoon"],
    index=0
)

jawg_token = "XnKSEzQMZxWKbrdxePWWk36HSJhZQyFVDuY1tMX5oDQ9LSpRUnfmW41sSLVJoC5h"

# Layout setup: Map Column (Left) and Details Column (Right)
col_map, col_details = st.columns([1.1, 0.9])

with col_map:
    st.subheader("🗺️ Mapa Interactivo de Incendios")
    
    # 1. Municipality filter selector above the map
    muni_options = ["Todos"] + sorted(gdf_m['MpNombre'].tolist())
    
    if st.session_state["selected_municipality"] in muni_options:
        default_idx = muni_options.index(st.session_state["selected_municipality"])
    else:
        default_idx = 0
        
    selected_muni = st.selectbox(
        "Filtrar por municipio:",
        options=muni_options,
        index=default_idx,
        key="muni_select"
    )
    
    # Update state and rerun if the selectbox changes
    if selected_muni != st.session_state["selected_municipality"]:
        st.session_state["selected_municipality"] = selected_muni
        st.rerun()
        
    # 2. Filter wildfires based on selection
    if st.session_state["selected_municipality"] != "Todos":
        active_wildfire_ids = joined[joined['MpNombre'] == st.session_state["selected_municipality"]]['id'].unique()
        gdf_w_filtered = gdf_w[gdf_w['id'].isin(active_wildfire_ids)]
    else:
        gdf_w_filtered = gdf_w

    # Center coordinates and zoom defaults
    map_center = [10.65, -74.95]
    zoom_start = 9.5
    
    # Configure folium map with the selected style
    if map_style == "CartoDB Positron":
        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB positron")
    elif map_style == "CartoDB Dark Matter":
        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles="CartoDB dark_matter")
    elif map_style == "Jawg Lagoon":
        tile_url = f"https://tile.jawg.io/jawg-lagoon/{{z}}/{{x}}/{{y}}{{r}}.png?access-token={jawg_token}"
        attribution = '&copy; <a href="https://www.jawg.io" target="_blank">Jawg</a> &copy; <a href="https://www.openstreetmap.org/copyright" target="_blank">OpenStreetMap</a>'
        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=tile_url, attr=attribution)
    else: # Esri World Imagery
        esri_url = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'
        esri_attr = 'Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community'
        m = folium.Map(location=map_center, zoom_start=zoom_start, tiles=esri_url, attr=esri_attr)
    
    # Dynamically fly the map view to the bounds of the selection
    if st.session_state["selected_municipality"] != "Todos":
        muni_row = gdf_m[gdf_m['MpNombre'] == st.session_state["selected_municipality"]]
        if not muni_row.empty:
            bounds = muni_row.geometry.iloc[0].bounds
            m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
    else:
        bounds = gdf_m.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        
    # Municipality styling functions
    def style_fn(feature):
        muni_name = feature['properties']['MpNombre']
        is_selected = (muni_name == st.session_state["selected_municipality"])
        return {
            'fillColor': '#eab308' if is_selected else '#ffff00',
            'color': '#facc15', # Bright yellow border for high contrast against satellite and dark basemaps
            'weight': 3.0 if is_selected else 1.5,
            'fillOpacity': 0.20 if is_selected else 0.02
        }
        
    def highlight_fn(feature):
        return {
            'weight': 3.5,
            'color': '#facc15',
            'fillOpacity': 0.30
        }
        
    # Render Municipality layer
    folium.GeoJson(
        gdf_m,
        style_function=style_fn,
        highlight_function=highlight_fn,
        tooltip=folium.GeoJsonTooltip(
            fields=['MpNombre'],
            aliases=['Municipio:'],
            localize=True
        )
    ).add_to(m)
    
    # Render Wildfire layer (Polygons styled in bright red/orange)
    def wildfire_style_fn(feature):
        return {
            'fillColor': '#f97316',
            'color': '#ea580c',
            'weight': 1.5,
            'fillOpacity': 0.65
        }
        
    if not gdf_w_filtered.empty:
        folium.GeoJson(
            gdf_w_filtered,
            style_function=wildfire_style_fn,
            tooltip=folium.GeoJsonTooltip(
                fields=['oldest_acquisition', 'area', 'fire_confidence'],
                aliases=['Foco Detectado:', 'Área (ha):', 'Confianza:'],
                localize=True
            )
        ).add_to(m)
        
    # Render Folium Map
    map_data = st_folium(m, height=520, use_container_width=True, key="folium_map")
    
    # Map click handling: update selection state when a municipality is clicked
    if map_data and map_data.get("last_active_drawing"):
        clicked_properties = map_data["last_active_drawing"].get("properties")
        if clicked_properties and "MpNombre" in clicked_properties:
            clicked_muni = clicked_properties["MpNombre"]
            if clicked_muni != st.session_state["selected_municipality"]:
                st.session_state["selected_municipality"] = clicked_muni
                st.rerun()

with col_details:
    # 3. Details & Metrics Section
    selected = st.session_state["selected_municipality"]
    
    if selected != "Todos":
        muni_row = gdf_m[gdf_m['MpNombre'] == selected].iloc[0]
        area_sq_m = muni_row['area_m2']
        perimeter_m = muni_row['perimeter_m']
        
        muni_wildfires = joined[joined['MpNombre'] == selected]
        total_fires = len(muni_wildfires)
        title_text = selected
        subtitle_text = "Municipio"
    else:
        area_sq_m = dept_area
        perimeter_m = dept_perimeter
        total_fires = len(gdf_w)
        title_text = "Departamento del Atlántico"
        subtitle_text = "Departamento"
        
    st.markdown(f"""
        <div style="margin-bottom: 15px;">
            <span style="font-size: 13px; text-transform: uppercase; color: #f97316; font-weight: 600; letter-spacing: 0.05em;">{subtitle_text}</span>
            <h2 style="margin: 0; font-size: 28px; line-height: 1.2; color: #0f172a;">{title_text}</h2>
        </div>
    """, unsafe_allow_html=True)
    
    # Metrics Cards
    col_m1, col_m2, col_m3 = st.columns(3)
    with col_m1:
        st.markdown(f"""
            <div class="metric-card">
                <div class="card-title">Área Total</div>
                <div class="card-value">{area_sq_m:,.0f}</div>
                <div class="card-subtitle">metros cuadrados (m²)</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m2:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3b82f6;">
                <div class="card-title">Perímetro</div>
                <div class="card-value">{perimeter_m:,.0f}</div>
                <div class="card-subtitle">metros (m)</div>
            </div>
        """, unsafe_allow_html=True)
    with col_m3:
        st.markdown(f"""
            <div class="metric-card" style="border-left-color: #f97316;">
                <div class="card-title">Incendios</div>
                <div class="card-value">{total_fires:,}</div>
                <div class="card-subtitle">focos registrados</div>
            </div>
        """, unsafe_allow_html=True)
        
    # Reset button if a specific municipality is selected
    if selected != "Todos":
        if st.button("🌐 Mostrar Todo el Departamento", use_container_width=True):
            st.session_state["selected_municipality"] = "Todos"
            st.rerun()
            
        # Contextual Info
        pct = (total_fires / len(gdf_w)) * 100 if len(gdf_w) > 0 else 0
        st.markdown(f"""
            <div style="font-size: 13px; color: #475569; background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 15px;">
                El municipio de <b>{selected}</b> concentra el <b>{pct:.1f}%</b> de los incendios forestales totales del Atlántico ({total_fires} de {len(gdf_w)}).
            </div>
        """, unsafe_allow_html=True)

    # 4. Bar Plot: Wildfires by Municipality
    all_munis = sorted(gdf_m['MpNombre'].unique())
    muni_counts = pd.Series(0, index=all_munis)
    counts = joined['MpNombre'].value_counts()
    muni_counts.update(counts)
    
    df_muni_plot = muni_counts.reset_index(name='count')
    df_muni_plot.rename(columns={'index': 'MpNombre'}, inplace=True)
    
    # Sort ascending for horizontal bar chart (so the highest count renders at the top)
    df_muni_plot = df_muni_plot.sort_values(by='count', ascending=True)
    
    # Set highlight colors: selected is red-orange, all others are gray. If none is selected, all are red-orange.
    if selected != "Todos":
        bar_colors = [
            '#ef4444' if row['MpNombre'] == selected else '#cbd5e1'
            for _, row in df_muni_plot.iterrows()
        ]
    else:
        bar_colors = ['#f97316'] * len(df_muni_plot)
        
    fig_muni = go.Figure()
    fig_muni.add_trace(go.Bar(
        y=df_muni_plot['MpNombre'],
        x=df_muni_plot['count'],
        orientation='h',
        marker_color=bar_colors,
        hovertemplate="<b>%{y}</b><br>Incendios: %{x}<extra></extra>"
    ))
    
    fig_muni.update_layout(
        title=dict(
            text="Incendios Forestales por Municipio",
            font=dict(family="Outfit, sans-serif", size=15, color="#1e293b", weight=700)
        ),
        xaxis=dict(title=dict(text="Número de Incendios", font=dict(size=11)), gridcolor="#f1f5f9"),
        yaxis=dict(title="", tickfont=dict(size=9)),
        template="plotly_white",
        margin=dict(l=10, r=10, t=35, b=10),
        height=520
    )
    st.plotly_chart(fig_muni, use_container_width=True)
    
    # 5. Bar Plot: Wildfires by Month
    all_months = sorted(gdf_w['year_month'].unique())
    month_counts = pd.Series(0, index=all_months)
    
    if selected != "Todos":
        filtered_joined = joined[joined['MpNombre'] == selected]
        m_counts = filtered_joined['year_month'].value_counts()
    else:
        m_counts = gdf_w['year_month'].value_counts()
        
    month_counts.update(m_counts)
    df_month_plot = month_counts.reset_index(name='count')
    df_month_plot.rename(columns={'index': 'year_month'}, inplace=True)
    
    # Map to nice Spanish labels
    SPANISH_MONTH_FULL = {
        '2025-12': 'Dic 2025',
        '2026-01': 'Ene 2026',
        '2026-02': 'Feb 2026',
        '2026-03': 'Mar 2026',
        '2026-04': 'Abr 2026',
        '2026-05': 'May 2026',
        '2026-06': 'Jun 2026',
        '2026-07': 'Jul 2026',
    }
    
    def get_month_label(ym_str):
        if ym_str in SPANISH_MONTH_FULL:
            return SPANISH_MONTH_FULL[ym_str]
        try:
            parts = ym_str.split('-')
            year = parts[0]
            month_num = int(parts[1])
            months = ['Ene', 'Feb', 'Mar', 'Abr', 'May', 'Jun', 'Jul', 'Ago', 'Sep', 'Oct', 'Nov', 'Dic']
            return f"{months[month_num-1]} {year}"
        except:
            return ym_str
            
    df_month_plot['month_label'] = df_month_plot['year_month'].apply(get_month_label)
    
    fig_month = go.Figure()
    fig_month.add_trace(go.Bar(
        x=df_month_plot['month_label'],
        y=df_month_plot['count'],
        marker_color='#ef4444',
        hovertemplate="<b>%{x}</b><br>Incendios: %{y}<extra></extra>"
    ))
    
    fig_month.update_layout(
        title=dict(
            text="Distribución Temporal (Incendios por Mes)",
            font=dict(family="Outfit, sans-serif", size=15, color="#1e293b", weight=700)
        ),
        xaxis=dict(title="", tickfont=dict(size=10)),
        yaxis=dict(title=dict(text="Número de Incendios", font=dict(size=11)), gridcolor="#f1f5f9"),
        template="plotly_white",
        margin=dict(l=10, r=10, t=35, b=10),
        height=280
    )
    st.plotly_chart(fig_month, use_container_width=True)
