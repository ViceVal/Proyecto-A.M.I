import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# --- DATOS DEL PROYECTO ---
CUERDAS_GUITARRA = {
    "E2 (Sexta)": (161.50, 150, 175),
    "A2 (Quinta)": (220.70, 95, 125), 
    "D3 (Cuarta)": (146.83, 130, 155),
    "G3 (Tercera)": (196.00, 180, 220), 
    "B3 (Segunda)": (246.94, 230, 270), 
    "E4 (Primera)": (329.63, 300, 350)
}

# --- DATOS TEÓRICOS REALES ---
FRECUENCIAS_ESTANDAR = {
    "E2 (Sexta)": 82.41,
    "A2 (Quinta)": 110.00,
    "D3 (Cuarta)": 146.83,
    "G3 (Tercera)": 196.00,
    "B3 (Segunda)": 246.94,
    "E4 (Primera)": 329.63
}

# --- DICCIONARIOS DE TRADUCCIÓN ---
NOTAS_LATINAS = {
    "C": "Do", "C#": "Do#", "D": "Re", "D#": "Re#", 
    "E": "Mi", "F": "Fa", "F#": "Fa#", "G": "Sol", 
    "G#": "Sol#", "A": "La", "A#": "La#", "B": "Si"
}

ACORDES_GUITARRA = {
    "E Mayor": ["E", "G#", "B"], "E menor": ["E", "G", "B"],
    "A Mayor": ["A", "C#", "E"], "A menor": ["A", "C", "E"],
    "D Mayor": ["D", "F#", "A"], "D menor": ["D", "F", "A"],
    "G Mayor": ["G", "B", "D"],  "G menor": ["G", "A#", "D"],
    "C Mayor": ["C", "E", "G"],  "C menor": ["C", "D#", "G"],
    "F Mayor": ["F", "A", "C"],  "F menor": ["F", "G#", "C"],
    "B Mayor": ["B", "D#", "F#"],"B menor": ["B", "D", "F#"],
}

# --- FUNCIONES ---
def ingles_a_latino(nota_ingles):
    nota_pura = ''.join([c for c in nota_ingles if not c.isdigit()])
    return NOTAS_LATINAS.get(nota_pura, nota_ingles)

# --- GRÁFICAS ---

def mostrar_grafica_afinador():
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    
    nombres_originales = list(CUERDAS_GUITARRA.keys())
    y_pos = np.arange(len(nombres_originales))
    nuevas_etiquetas_y = [] 
    
    colores = ['#e67e22', '#c0392b', '#f1c40f', '#27ae60', '#9b59b6', '#2980b9']
    COLOR_TEORICO = '#d35400' 
    COLOR_TARGET = '#2c3e50'
    
    ax.set_title('Precisión de Afinación: Configuración Actual vs. Estándar Real (A4=440Hz)', fontsize=16, pad=20, color='#2c3e50', weight='bold')

    for i, (nombre_cuerda, datos) in enumerate(CUERDAS_GUITARRA.items()):
        objetivo_codigo, min_f, max_f = datos
        ancho = max_f - min_f
        
        # 1. Obtener la Frecuencia Real Estándar directamente del diccionario corregido
        freq_teorica_val = FRECUENCIAS_ESTANDAR.get(nombre_cuerda, 0.0)
        
        # Extraer nota base (E, A, D...) para traducción
        nota_base = nombre_cuerda.split(" ")[0][0] # Toma la primera letra
        nota_latina = NOTAS_LATINAS.get(nota_base, nota_base)
        
        # 2. Construir Etiqueta Y
        partes = nombre_cuerda.split("(")
        nombre_tecnico = partes[0].strip()
        nombre_vulgar = partes[1].replace(")", "")
        etiqueta_final = f"{nombre_tecnico} ({nota_latina})\n{nombre_vulgar}"
        nuevas_etiquetas_y.append(etiqueta_final)
        
        # 3. Dibujar Rango (Fondo)
        rect = patches.Rectangle((min_f, i - 0.25), ancho, 0.5, 
                               linewidth=0, facecolor=colores[i], alpha=0.2)
        ax.add_patch(rect)
        
        # 4. Línea de discrepancia
        # Si la diferencia es muy grande (como en E2), la línea será larga
        ax.plot([objetivo_codigo, freq_teorica_val], [i, i], color='#95a5a6', linestyle='--', linewidth=1, alpha=0.8)
        
        # 5. Marcadores
        ax.plot([objetivo_codigo, objetivo_codigo], [i-0.3, i+0.3], color=colores[i], linewidth=3)
        ax.plot(freq_teorica_val, i, 'o', color=COLOR_TEORICO, markersize=8)
        
        # 6. Textos Inteligentes (Con fondo para no superponerse)
        props_obj = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='#bdc3c7', pad=0.2)
        props_teo = dict(boxstyle='round', facecolor='#fff3e0', alpha=0.8, edgecolor='#ffcc80', pad=0.2)

        # Target (Arriba)
        ax.text(objetivo_codigo, i + 0.35, f"Obj: {objetivo_codigo}", ha='center', va='bottom', 
                fontsize=9, color=COLOR_TARGET, weight='bold', bbox=props_obj)
        
        # Teórico (Abajo)
        ax.text(freq_teorica_val, i - 0.35, f"Teó: {freq_teorica_val:.1f}", ha='center', va='top', 
                fontsize=9, color=COLOR_TEORICO, style='italic', bbox=props_teo)

    # Configuración Ejes
    ax.set_yticks(y_pos)
    ax.set_yticklabels(nuevas_etiquetas_y, fontsize=11, fontweight='bold', color='#34495e')
    
    ax.set_xlabel('Frecuencia (Hz)', fontsize=10)
    
    # Ampliamos el rango X para que quepa el E2 real (82Hz) y el E4 (330Hz)
    ax.set_xlim(70, 360) 
    
    # Limpieza
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.grid(axis='y', visible=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # Leyenda
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#7f8c8d', lw=3),
                    Line2D([0], [0], marker='o', color='w', markerfacecolor=COLOR_TEORICO, markersize=8)]
    ax.legend(custom_lines, ['Objetivo Configurado', 'Estándar Real (Física)'], 
              loc='upper center', bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    
    plt.tight_layout()
    plt.show(block=False)

def mostrar_grafica_acordes():
    """Gráfica de Acordes sin cambios"""
    plt.style.use('seaborn-v0_8-whitegrid')
    
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    
    acordes_nombres = list(ACORDES_GUITARRA.keys())
    
    # Eje X Latino
    notas_ingles = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    notas_latinas = [ingles_a_latino(n) for n in notas_ingles]
    
    matriz = np.zeros((len(acordes_nombres), len(notas_ingles)))
    
    for i, acorde in enumerate(acordes_nombres):
        notas = ACORDES_GUITARRA[acorde]
        for nota in notas:
            nota_limpia = ''.join([c for c in nota if not c.isdigit()])
            if nota_limpia in notas_ingles:
                idx = notas_ingles.index(nota_limpia)
                matriz[i, idx] = 1

    cax = ax.imshow(matriz, cmap='Blues', aspect='auto', vmin=0, vmax=1.8)
    
    ax.set_xticks(np.arange(len(notas_latinas)))
    ax.set_xticklabels(notas_latinas, fontsize=12, fontweight='bold')
    ax.xaxis.tick_top() 
    
    ax.set_yticks(np.arange(len(acordes_nombres)))
    ax.set_yticklabels(acordes_nombres, fontsize=11)
    
    ax.set_xticks(np.arange(len(notas_latinas)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(acordes_nombres)) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    
    ax.set_title('Composición de Acordes (Notas Requeridas)', fontsize=15, y=-0.05, color='#2c3e50', weight='bold')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("Generando Gráficas...")
    mostrar_grafica_afinador()
    mostrar_grafica_acordes()