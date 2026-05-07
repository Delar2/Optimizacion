# 🚀 DESPLEGAR EN STREAMLIT CLOUD

## 📋 Resumen

Esta guía te llevará a tener tu app corriendo en Streamlit Cloud en **10 minutos**.

La app se ejecutará en los servidores de Streamlit (gratis) y podrás compartir un link público con tu equipo.

---

## 🎯 REQUISITOS

- ✅ Cuenta de GitHub (gratis en github.com)
- ✅ Cuenta de Streamlit (gratis en streamlit.io - con GitHub)
- ✅ Los 2 archivos que ya tienes:
  - `streamlit_optimizer.py`
  - `requirements.txt`

---

## 📦 PASO 1: Crear Repositorio en GitHub (2 minutos)

### 1.1 Ir a GitHub
- Ve a https://github.com/new
- Inicia sesión (crea cuenta si no tienes)

### 1.2 Crear Repositorio
- **Repository name:** `optimizador-muestreo` (o el nombre que quieras)
- **Description:** "Optimizador de estrategia de muestreo geológico"
- **Public** (importante para Streamlit Cloud)
- Haz clic en **"Create repository"**

### 1.3 Subir Archivos a GitHub

**Opción A: Subir directamente en GitHub (más fácil)**
1. En tu nuevo repo, haz clic en **"Add file"** → **"Upload files"**
2. Sube:
   - `streamlit_optimizer.py`
   - `requirements.txt`
3. Haz clic en **"Commit changes"**

**Opción B: Usar Git en tu computadora (para usuarios avanzados)**
```bash
git clone https://github.com/TU_USUARIO/optimizador-muestreo.git
cd optimizador-muestreo
# Copia los archivos aquí
cp /ruta/streamlit_optimizer.py .
cp /ruta/requirements.txt .
git add .
git commit -m "Initial commit"
git push origin main
```

### Resultado
Tu repositorio en GitHub ahora tiene:
```
optimizador-muestreo/
├── streamlit_optimizer.py
└── requirements.txt
```

---

## 🌐 PASO 2: Conectar en Streamlit Cloud (3 minutos)

### 2.1 Ir a Streamlit Cloud
- Ve a https://share.streamlit.io
- Haz clic en **"Sign in with GitHub"**
- Autoriza Streamlit a acceder a GitHub

### 2.2 Crear Nueva App
- Haz clic en **"New app"** (botón azul arriba a la derecha)

### 2.3 Configurar la App
Rellena los campos:

| Campo | Valor |
|-------|-------|
| **Repository** | `TU_USUARIO/optimizador-muestreo` |
| **Branch** | `main` (default) |
| **Main file path** | `streamlit_optimizer.py` |

### 2.4 Deploy
- Haz clic en **"Deploy!"**
- Espera 2-3 minutos mientras Streamlit:
  - Clona tu repositorio
  - Instala dependencias
  - Inicia la app

### Resultado
¡Tu app está en línea! Verás algo como:
```
https://optimizador-muestreo-abc123.streamlit.app
```

---

## ✅ PASO 3: Verificar que Funciona (2 minutos)

1. Abre tu URL pública en el navegador
2. Espera a que cargue (primera vez tarda ~30 segundos)
3. Verifica que ves:
   - ✅ Título "🎯 Optimizador de Estrategia de Muestreo"
   - ✅ Parámetros de entrada
   - ✅ Botón "▶ Resolver Modelo MILP"
4. Prueba hacer clic en "Resolver"
5. Espera a que termine la optimización
6. Verifica que ves resultados y gráficos

---

## 🎁 PASO 4: Compartir con tu Equipo (1 minuto)

Tu URL es pública. Puedes compartirla:
- 📧 Por email: `https://optimizador-muestreo-abc123.streamlit.app`
- 💬 Por Slack/Teams
- 📱 Por WhatsApp
- 🔗 En documentos

**Otros pueden:**
- ✅ Acceder sin instalar nada
- ✅ Usarla desde cualquier dispositivo
- ✅ Cambiar parámetros
- ✅ Descargar resultados en Excel

---

## 🔄 ACTUALIZAR LA APP (Si cambias el código)

Si necesitas modificar la app:

### 1. Editar en GitHub
- Ve a tu repositorio
- Edita `streamlit_optimizer.py`
- Haz commit de los cambios

### 2. Redeploy Automático
- Streamlit detecta automáticamente los cambios
- La app se actualiza en 1-2 minutos
- No necesitas hacer nada más

### O Redeploy Manual
- Ve a https://share.streamlit.io
- Busca tu app
- Haz clic en los "..." (menú)
- Selecciona **"Reboot app"**

---

## 🔧 TROUBLESHOOTING

### "La app tarda mucho en cargar"
- Normal en primer acceso (instala dependencias)
- Luego carga en 10-20 segundos
- Si tarda >5 minutos, ve a "Advanced settings" → aumenta memory

### "Error: ModuleNotFoundError"
- Verifica que `requirements.txt` está en el repo
- Verifica que tiene las líneas correctas
- Redeploy la app

### "La app no ve mis cambios"
- GitHub tarda ~1 minuto en actualizar
- Streamlit tarda 1-2 minutos más en redeploy
- Actualiza tu navegador (Ctrl+F5)

### "Quiero cambiar presupuesto/tiempo"
1. Edita `streamlit_optimizer.py` en GitHub
2. Busca las líneas con valores fijos
3. Cambia el valor
4. Commit
5. La app se actualiza automáticamente en 1-2 minutos

### "Los gráficos no se ven"
- Problema raro pero puede pasar
- Ve a la app → menú (⋯) → "Reboot app"
- Si persiste, abre issue en GitHub

---

## 📊 MONITOREAR LA APP

En https://share.streamlit.io puedes ver:
- ✅ Estado de la app (online/offline)
- ✅ Última actualización
- ✅ Opción de reboot
- ✅ Logs (si algo falla)

---

## 💾 RESPALDAR CÓDIGO

Recomendado: Guarda una copia local de los archivos:
```bash
# En tu computadora
git clone https://github.com/TU_USUARIO/optimizador-muestreo.git
```

Ahora tienes:
- ✅ Copia en GitHub (públicamente disponible)
- ✅ Copia en tu computadora (local)
- ✅ App corriendo en Streamlit Cloud (en línea)

---

## 🎓 CONCEPTOS

### ¿Por qué Streamlit Cloud?
- **Gratis:** Hosted en servidores de Streamlit
- **Fácil:** Sin configurar servidores
- **Rápido:** Sube y listo en minutos
- **Escalable:** Maneja múltiples usuarios

### ¿Qué sucede al hacer deploy?
1. Streamlit clona tu repositorio de GitHub
2. Instala dependencias (`requirements.txt`)
3. Inicia la app (`streamlit_optimizer.py`)
4. La ejecuta en sus servidores
5. Genera URL pública
6. Otros pueden acceder

### ¿Cómo se actualizan los cambios?
1. Editas el archivo en GitHub
2. Haces commit/push
3. Streamlit detecta cambios
4. Automáticamente reinicia la app
5. Tu cambio está en línea en 1-2 minutos

---

## ✨ VENTAJAS DE STREAMLIT CLOUD

| Aspecto | Beneficio |
|---------|-----------|
| **Costo** | Gratis (sin tarjeta de crédito) |
| **Hosting** | Servidores de Streamlit (confiable) |
| **Actualización** | Automática al hacer push a GitHub |
| **URL** | Pública y compartible |
| **Usuarios** | Sin límite de acceso concurrente |
| **Datos** | Se descargan localmente (privado) |

---

## 🎯 FLUJO COMPLETO

```
Tu Computadora
     ↓ (Creas archivos)
GitHub (Repositorio)
     ↓ (Conectas)
Streamlit Cloud (Deploy)
     ↓ (Generas URL)
Tu Equipo (Accede)
```

---

## 🚀 CHECKLIST FINAL

- [ ] Creé repositorio en GitHub
- [ ] Subí `streamlit_optimizer.py`
- [ ] Subí `requirements.txt`
- [ ] Conecté GitHub a Streamlit Cloud
- [ ] La app está deployada
- [ ] Probé que funciona
- [ ] Compartí URL con mi equipo
- [ ] Otros pueden acceder sin problemas

---

## 📞 SOPORTE

Si algo no funciona:

1. **Verifica:** ¿Están todos los archivos en GitHub?
2. **Revisa:** ¿Tiene `requirements.txt` todas las librerías?
3. **Comprueba:** ¿Es el repositorio público?
4. **Intenta:** Reboot de la app en Streamlit Cloud
5. **Lee:** https://docs.streamlit.io/deploy/streamlit-cloud

---

## 📈 PRÓXIMOS PASOS

1. ✅ Deploy completado
2. 📊 Usa la app en línea
3. 📝 Comparte con tu equipo
4. 🔄 Itera: cambia parámetros, actualiza código
5. 📥 Descarga resultados en Excel

---

**¡Tu app está lista para compartir con el mundo!** 🌍

Versión 1.0 | 2024
