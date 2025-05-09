import cv2
import streamlit as st
import numpy as np
import pandas as pd
import torch

# Configuración de página Streamlit
st.set_page_config(
    page_title="Detección de Objetos en Tiempo Real",
    page_icon="🔍",
    layout="wide"
)

# Función para cargar el modelo YOLOv5
@st.cache_resource
def load_yolov5_model():
    try:
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model
    except Exception as e:
        st.error(f"❌ Error al cargar el modelo: {str(e)}")
        st.info("""
        Recomendaciones:
        1. Asegúrate de tener una conexión a internet.
        2. Usa las versiones compatibles de torch y torchvision en tu requirements.txt:
           torch==1.12.0
           torchvision==0.13.0
        """)
        return None

# Título y descripción de la aplicación
st.title("🔍 Detección de Objetos en Imágenes")
st.markdown("""
Esta aplicación utiliza YOLOv5 para detectar objetos en imágenes capturadas con tu cámara.
Ajusta los parámetros en la barra lateral para personalizar la detección.
""")

# Cargar el modelo
with st.spinner("Cargando modelo YOLOv5..."):
    model = load_yolov5_model()

# Si el modelo se cargó correctamente, configuramos los parámetros
if model:
    # Sidebar para los parámetros de configuración
    st.sidebar.title("Parámetros")
    
    # Ajustar parámetros del modelo
    with st.sidebar:
        st.subheader('Configuración de detección')
        conf_thres = st.slider('Confianza mínima', 0.0, 1.0, 0.25, 0.01)
        iou_thres = st.slider('Umbral IoU', 0.0, 1.0, 0.45, 0.01)
        st.caption(f"Confianza: {conf_thres:.2f} | IoU: {iou_thres:.2f}")
    
    # Contenedor principal para la cámara y resultados
    main_container = st.container()
    
    with main_container:
        # Capturar foto con la cámara
        picture = st.camera_input("Capturar imagen", key="camera")
        
        if picture:
            # Procesar la imagen capturada
            bytes_data = picture.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Realizar la detección
            with st.spinner("Detectando objetos..."):
                try:
                    results = model(cv2_img, size=640)
                    results = results.pandas().xyxy[0]
                except Exception as e:
                    st.error(f"Error durante la detección: {str(e)}")
                    st.stop()
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Imagen con detecciones")
                img_with_boxes = model(cv2_img)
                img_with_boxes.render()
                st.image(img_with_boxes.ims[0], channels="BGR", use_column_width=True)
            
            with col2:
                st.subheader("Objetos detectados")
                if not results.empty:
                    results_filtered = results[results['confidence'] >= conf_thres]
                    st.dataframe(results_filtered[['name', 'confidence']], use_container_width=True)
                    category_counts = results_filtered['name'].value_counts()
                    st.bar_chart(category_counts)
                else:
                    st.info("No se detectaron objetos con los parámetros actuales.")
                    st.caption("Prueba a reducir el umbral de confianza en la barra lateral.")
else:
    st.error("No se pudo cargar el modelo. Por favor verifica las dependencias e inténtalo nuevamente.")
    st.stop()

# Información adicional y pie de página
st.markdown("---")
st.caption("""
**Acerca de la aplicación**: Esta aplicación utiliza YOLOv5 para detección de objetos en tiempo real.
Desarrollada con Streamlit y PyTorch.
""")
