>>> # 1. Contexto y Origen de los datos 

>>> # Descripción de los datos
>>> # Base de datos de la USGS (United States Geological Survey), 
>>> # Nivel del río Misisipi 
>>> # Punto de recolección de los datos es el "Mississippi River at Thebes, IL - 07022000" 
>>> # Serie original de 366,082 datos
>>> # Frecuencia temporal irregular, que se ajustó a una frecuencia de 30 minutos 
>>> # Los datos se miden en pies y cubren el periodo desde 1995 hasta 2025

>>> # 2. Análisis preliminar

>>> import pandas as pd
>>>
>>> # Ruta del archivo original en WSL
>>> file_path = '/mnt/c/Users/Yuliceth Ramos/datos_Mississippi.csv'
>>>
>>> # Cargar el archivo CSV
>>> df = pd.read_csv(file_path)
>>>
>>> # Convertir la columna 'datetime' al formato adecuado
>>> df['datetime'] = pd.to_datetime(df['datetime'], format='%m/%d/%Y %H:%M')
>>> # Establecer la columna 'datetime' como índice
>>> df.set_index('datetime', inplace=True)
>>>
>>> # Verificar las primeras filas del DataFrame
>>> print(df.head())
                 gage_height
datetime
3/02/1995 14:00       212000
3/02/1995 14:21       211000
3/02/1995 15:00       211000
3/02/1995 15:21       211000
3/02/1995 16:00       211000
>>> # Ajustar la frecuencia de la serie temporal a 30 minutos (resampling)
>>> df_resampled = df.resample('30min').mean()
>>> # Guardar el archivo ajustado con los datos resampleados
>>> output_file_path = '/mnt/c/Users/Yuliceth Ramos/datos_Mississippi_adjusted.csv'
>>> df_resampled.to_csv(output_file_path)
>>> import pandas as pd
>>>
>>> # Ruta del archivo ajustado
>>> file_path = '/mnt/c/Users/Yuliceth Ramos/datos_Mississippi_adjusted.csv'
>>>
>>> # Cargar el archivo CSV
>>> df = pd.read_csv(file_path)
>>> # Ver los primeros valores del archivo ajustado
>>> print("\nPrimeros valores del archivo ajustado:")

Primeros valores del archivo ajustado:
>>> print(df.head())
                 gage_height
datetime
3/02/1995 14:00     211500.0
3/02/1995 14:30          NaN
3/02/1995 15:00     211000.0
3/02/1995 15:30          NaN
3/02/1995 16:00     211500.0

>>> # Imputar los valores faltantes utilizando medias móviles
>>> df_resampled['gage_height'] = df_resampled['gage_height'].fillna(df_resampled['gage_height'].rolling(window=5, min_periods=1).mean())
>>> import pandas as pd
>>>
>>> # Ruta del archivo ajustado con imputación de datos faltantes
>>> file_path = '/mnt/c/Users/Yuliceth Ramos/datos_Mississippi_adjusted_filled.csv'
>>>
>>> # Cargar el archivo CSV
>>> df = pd.read_csv(file_path)

>>> # Verificar si hay datos faltantes después de la imputación
>>> missing_data = df.isnull().sum()
>>>
>>> # Mostrar los resultados
>>> print("Datos faltantes por columna:")
Datos faltantes por columna:
>>> print(missing_data)
gage_height           16625
gage_height_filled    16625
dtype: int64

>>> # Imputar los valores faltantes utilizando interpolación lineal
>>> df['gage_height_filled'] = df['gage_height'].interpolate(method='linear')
>>> # Verificar si los datos faltantes se han imputado correctamente
>>> print(f"Datos faltantes después de imputar con interpolación: \n{df.isnull().sum()}")
Datos faltantes después de imputar con interpolación:
gage_height           16625
gage_height_filled        0
dtype: int64

>>> import pandas as pd
>>> import numpy as np
>>> import matplotlib.pyplot as plt
>>> from statsmodels.tsa.seasonal import seasonal_decompose
>>> # Ruta al archivo en WSL
>>> file_path = '/mnt/c/Users/Yuliceth Ramos/datos_Mississippi_filled.csv'
>>> # Cargar los datos
>>> df = pd.read_csv(file_path)
>>> # Verificar las primeras filas
>>> print(df.head())
              datetime  gage_height
0  1995-02-03 14:00:00     211500.0
1  1995-02-03 14:30:00     211500.0
2  1995-02-03 15:00:00     211000.0
3  1995-02-03 15:30:00     211250.0
4  1995-02-03 16:00:00     211500.0
>>> # Obtener el resumen estadístico de la columna 'gage_height'
>>> summary_stats = df['gage_height'].describe()
>>> print("\nResumen estadístico de la serie temporal:")

Resumen estadístico de la serie temporal:
>>> print(summary_stats)
count    5.255990e+05
mean     2.504042e+05
std      1.536288e+05
min      0.000000e+00
25%      1.290000e+05
50%      2.030000e+05
75%      3.360000e+05
max      1.050000e+06
Name: gage_height, dtype: float64
>>> # Calcular la mediana y desviación estándar
>>> median = df['gage_height'].median()
>>> std_dev = df['gage_height'].std()
>>> print(f"\nMediana de los datos: {median}")
Mediana de los datos: 203000.0
>>> print(f"Desviación estándar de los datos: {std_dev}")
Desviación estándar de los datos: 153628.8129807574

>>> # 3. Visualización de la serie

>>> # Convertir la columna datetime a formato de fecha
>>> df['datetime'] = pd.to_datetime(df['datetime'])
>>> # Graficar la serie temporal original
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> plt.plot(df['datetime'], df['gage_height'], label='Nivel del río', color='b')
[<matplotlib.lines.Line2D object at 0x7fc1187963d0>]
>>> plt.title('Serie Temporal: Nivel del Río Misisipi')
Text(0.5, 1.0, 'Serie Temporal: Nivel del Río Misisipi')
>>> plt.xlabel('Fecha')
Text(0.5, 0, 'Fecha')
>>> plt.ylabel('Nivel del río (en pies)')
Text(0, 0.5, 'Nivel del río (en pies)')
>>> plt.xticks(df['datetime'][::50000], df['datetime'].dt.year[::50000], rotation=45)
([<matplotlib.axis.XTick object at 0x7fc1187e2640>, <matplotlib.axis.XTick object at 0x7fc1187e2610>, <matplotlib.axis.XTick object at 0x7fc1187a3490>, <matplotlib.axis.XTick object at 0x7fc1187493a0>, <matplotlib.axis.XTick object at 0x7fc118749e50>, <matplotlib.axis.XTick object at 0x7fc11874f3a0>, <matplotlib.axis.XTick object at 0x7fc11874fe50>, <matplotlib.axis.XTick object at 0x7fc118757940>, <matplotlib.axis.XTick object at 0x7fc11875d430>, <matplotlib.axis.XTick object at 0x7fc118757ca0>, <matplotlib.axis.XTick object at 0x7fc11875de80>], [Text(9164.583333333334, 0, '1995'), Text(10206.25, 0, '1997'), Text(11247.916666666666, 0, '2000'), Text(12289.583333333334, 0, '2003'), Text(13331.25, 0, '2006'), Text(14372.916666666666, 0, '2009'), Text(15414.583333333334, 0, '2012'), Text(16456.25, 0, '2015'), Text(17497.916666666668, 0, '2017'), Text(18539.583333333332, 0, '2020'), Text(19581.25, 0, '2023')])
>>> plt.legend()
<matplotlib.legend.Legend object at 0x7fc1187a3b50>
>>> plt.grid(True)
>>> # Guardar la gráfica de la serie temporal
>>> output_file_path_serie = '/mnt/c/Users/Yuliceth Ramos/OneDrive/Documentos/DOCTORADO/4. ELECTIVA II - Series de Tiempo/serie_temporal.png'
>>> plt.tight_layout()
>>> plt.savefig(output_file_path_serie)
>>> plt.show()
>>> print(f"Gráfica de la serie temporal guardada en: {output_file_path_serie}")
Gráfica de la serie temporal guardada en: /mnt/c/Users/Yuliceth Ramos/OneDrive/Documentos/DOCTORADO/4. ELECTIVA II - Series de Tiempo/serie_temporal.png
>>> # Descomponer la serie temporal
>>> decomposition = seasonal_decompose(df['gage_height'], model='additive', period=50000)
>>> # Graficar la descomposición
>>> plt.figure(figsize=(10, 8))
<Figure size 1000x800 with 0 Axes>
>>> # Graficar la serie original
>>> plt.subplot(411)
<Axes: >
>>> plt.plot(df['datetime'], df['gage_height'], label='Original', color='b')
[<matplotlib.lines.Line2D object at 0x7fc1186f3310>]
>>> plt.legend(loc='upper left')
<matplotlib.legend.Legend object at 0x7fc1187c67f0>
>>> # Graficar la tendencia
>>> plt.subplot(412)
<Axes: >
>>> plt.plot(df['datetime'], decomposition.trend, label='Tendencia', color='r')
[<matplotlib.lines.Line2D object at 0x7fc118565f70>]
>>> plt.legend(loc='upper left')
<matplotlib.legend.Legend object at 0x7fc1188259d0>
>>> # Graficar la estacionalidad
>>> plt.subplot(413)
<Axes: >
>>> plt.plot(df['datetime'], decomposition.seasonal, label='Estacionalidad', color='g')
[<matplotlib.lines.Line2D object at 0x7fc118657d30>]
>>> plt.legend(loc='upper left')
<matplotlib.legend.Legend object at 0x7fc1186a8a30>
>>> # Graficar los residuos
>>> plt.subplot(414)
<Axes: >
>>> plt.plot(df['datetime'], decomposition.resid, label='Residuales', color='b')
[<matplotlib.lines.Line2D object at 0x7fc11866e460>]
>>> plt.legend(loc='upper left')
<matplotlib.legend.Legend object at 0x7fc1186a8a00>
>>> # Ajustar formato de la fecha para los ejes X
>>> for i in range(1, 5):
...     plt.subplot(4, 1, i)
...     plt.xticks(df['datetime'][::50000], df['datetime'].dt.year[::50000], rotation=45)
>>> # Guardar la gráfica de la descomposición
>>> output_file_path_decomposition = '/mnt/c/Users/Yuliceth Ramos/OneDrive/Documentos/DOCTORADO/4. ELECTIVA II - Series de Tiempo/descomposicion_temporal.png'
>>> plt.tight_layout()
>>> plt.savefig(output_file_path_decomposition)
>>> plt.show()
>>> print(f"Gráfica de la descomposición guardada en: {output_file_path_decomposition}")
Gráfica de la descomposición guardada en: /mnt/c/Users/Yuliceth Ramos/OneDrive/Documentos/DOCTORADO/4. ELECTIVA II - Series de Tiempo/descomposicion_temporal.png
>>> # 3.3. Detección de anomalías
>>> # Calcular Q1, Q3 y el IQR
>>> Q1 = df['gage_height_filled'].quantile(0.25)
>>> Q3 = df['gage_height_filled'].quantile(0.75)
>>> IQR = Q3 - Q1
>>> # Definir los límites inferior y superior
>>> lower_bound = Q1 - 1.5 * IQR
>>> upper_bound = Q3 + 1.5 * IQR
>>> # Filtrar los valores atípicos
>>> outliers_iqr = df[(df['gage_height_filled'] < lower_bound) | (df['gage_height_filled'] > upper_bound)]
>>> # Mostrar los valores atípicos
>>> print(f"Valores atípicos (IQR):\n{outliers_iqr}")
Valores atípicos (IQR):
                     gage_height  gage_height_filled
datetime
1995-05-18 05:00:00     651000.0            651000.0
1995-05-18 06:00:00     650000.0            650000.0
1995-05-18 06:30:00     650500.0            650500.0
1995-05-18 07:00:00     657000.0            657000.0
1995-05-18 07:30:00     653500.0            653500.0
...                          ...                 ...
2019-07-09 12:00:00     648000.0            648000.0
2019-07-09 12:30:00     647000.0            647000.0
2019-07-09 13:00:00     648000.0            648000.0
2019-07-09 14:30:00     647000.0            647000.0
2019-07-09 17:00:00     647000.0            647000.0

[10037 rows x 2 columns]

>>> # 4. Estacionalidad y periodicidad

>>> # ACF (Autocorrelación)
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plot_acf(df['gage_height'], lags=50, ax=plt.gca())
<Figure size 1200x600 with 1 Axes>
>>> plt.title('Autocorrelation Function (ACF) for Gage Height')
Text(0.5, 1.0, 'Autocorrelation Function (ACF) for Gage Height')
>>> plt.xlabel('Lags')
Text(0.5, 0, 'Lags')
>>> plt.ylabel('ACF')
Text(0, 0.5, 'ACF')
>>> plt.show()
>>> # PACF (Autocorrelación Parcial)
>>> plt.figure(figsize=(12, 6))
<Figure size 1200x600 with 0 Axes>
>>> plot_pacf(df['gage_height'], lags=50, ax=plt.gca())
<Figure size 1200x600 with 1 Axes>
>>> plt.title('Partial Autocorrelation Function (PACF) for Gage Height')
Text(0.5, 1.0, 'Partial Autocorrelation Function (PACF) for Gage Height')
>>> plt.xlabel('Lags')
Text(0.5, 0, 'Lags')
>>> plt.ylabel('PACF')
Text(0, 0.5, 'PACF')
>>> plt.show()
>>> # Diagramas de caja por periodo (año)
>>> import seaborn as sns
>>> # Convertir 'datetime' a formato de fecha
>>> df['datetime'] = pd.to_datetime(df['datetime'])
>>> # Extraer el año del datetime
>>> df['year'] = df['datetime'].dt.year
>>> # Crear el diagrama de cajas por periodo (año)
>>> plt.figure(figsize=(10,6))
<Figure size 1000x600 with 0 Axes>
>>> sns.boxplot(x='year', y='gage_height', data=df)
<Axes: xlabel='year', ylabel='gage_height'>
>>> # Añadir título y etiquetas
>>> plt.title('Distribución de gage_height_filled por Año')
Text(0.5, 1.0, 'Distribución de gage_height_filled por Año')
>>> plt.xlabel('Año')
Text(0.5, 0, 'Año')
>>> plt.ylabel('Nivel del río (en pies)')
Text(0, 0.5, 'Nivel del río (en pies)')
>>> plt.xticks(rotation=45)  # Rotar las etiquetas del eje x para mejorar la legibilidad
([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [Text(0, 0, '1995'), Text(1, 0, '1996'), Text(2, 0, '1997'), Text(3, 0, '1998'), Text(4, 0, '1999'), Text(5, 0, '2000'), Text(6, 0, '2001'), Text(7, 0, '2002'), Text(8, 0, '2003'), Text(9, 0, '2004'), Text(10, 0, '2005'), Text(11, 0, '2006'), Text(12, 0, '2007'), Text(13, 0, '2008'), Text(14, 0, '2009'), Text(15, 0, '2010'), Text(16, 0, '2011'), Text(17, 0, '2012'), Text(18, 0, '2013'), Text(19, 0, '2014'), Text(20, 0, '2015'), Text(21, 0, '2016'), Text(22, 0, '2017'), Text(23, 0, '2018'), Text(24, 0, '2019'), Text(25, 0, '2020'), Text(26, 0, '2021'), Text(27, 0, '2022'), Text(28, 0, '2023'), Text(29, 0, '2024'), Text(30, 0, '2025')])
>>> plt.tight_layout()
>>> # Mostrar la gráfica
>>> plt.show()
>>> # Transformación del dominio del tiempo
>>> import pandas as pd
>>> df['datetime'] = pd.to_datetime(df['datetime'])  # Convertir a formato datetime
>>> df.set_index('datetime', inplace=True)  # Establecer datetime como índice
>>> print(df.head())
                     gage_height  year
datetime
1995-02-03 14:00:00     211500.0  1995
1995-02-03 14:30:00     211500.0  1995
1995-02-03 15:00:00     211000.0  1995
1995-02-03 15:30:00     211250.0  1995
1995-02-03 16:00:00     211500.0  1995
>>> # Agrupar los datos por trimestre usando resample
>>> df_quarterly = df.resample('Q').mean()  # 'Q' es para trimestral
<stdin>:1: FutureWarning: 'Q' is deprecated and will be removed in a future version, please use 'QE' instead.
>>> # Ver los primeros valores del agrupamiento trimestral
>>> print(df_quarterly.head())
              gage_height    year
datetime
1995-03-31  188692.669867  1995.0
1995-06-30  509990.327381  1995.0
1995-09-30  236115.375906  1995.0
1995-12-31  156009.393871  1995.0
1996-03-31  141259.100275  1996.0
>>> # Graficar los datos trimestrales
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> df_quarterly['gage_height'].plot()
<Axes: xlabel='datetime'>
>>> plt.title('Nivel del río Misisipi - Promedio Trimestral')
Text(0.5, 1.0, 'Nivel del río Misisipi - Promedio Trimestral')
>>> plt.xlabel('Fecha')
Text(0.5, 0, 'Fecha')
>>> plt.ylabel('Nivel del río (en pies)')
Text(0, 0.5, 'Nivel del río (en pies)')
>>> plt.grid(True)
>>> plt.tight_layout()
>>> plt.show()

>>> # 5. Análisis de tendencia

>>> # suavizacion exponencial
>>> df['smoothed_exponential'] = df['gage_height'].ewm(span=12, adjust=False).mean()
>>> # Graficar los datos originales y los suavizados
>>> import matplotlib.pyplot as plt
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> plt.plot(df['gage_height'], label='Datos Originales', color='blue')
[<matplotlib.lines.Line2D object at 0x7fcae8dc53a0>]
>>> plt.plot(df['smoothed_exponential'], label='Suavizado Exponencial', color='red')
[<matplotlib.lines.Line2D object at 0x7fcae8df37c0>]
>>> plt.title('Nivel del Río Misisipi - Suavizamiento Exponencial')
Text(0.5, 1.0, 'Nivel del Río Misisipi - Suavizamiento Exponencial')
>>> plt.xlabel('Fecha')
Text(0.5, 0, 'Fecha')
>>> plt.ylabel('Nivel del río (en pies)')
Text(0, 0.5, 'Nivel del río (en pies)')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x7fcae8df3eb0>
>>> plt.grid(True)
>>> plt.tight_layout()
>>> plt.show()
>>> # Media móvil simple con una ventana de 12 períodos
>>> df['smoothed_moving_average'] = df['gage_height'].rolling(window=12).mean()
>>> # Graficar los datos originales y la media móvil
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> plt.plot(df['gage_height'], label='Datos Originales', color='blue')
[<matplotlib.lines.Line2D object at 0x7fcae8d27460>]
>>> plt.plot(df['smoothed_moving_average'], label='Media Móvil', color='green')
[<matplotlib.lines.Line2D object at 0x7fcae8d1cbb0>]
>>> plt.title('Nivel del Río Misisipi - Media Móvil')
Text(0.5, 1.0, 'Nivel del Río Misisipi - Media Móvil')
>>> plt.xlabel('Fecha')
Text(0.5, 0, 'Fecha')
>>> plt.ylabel('Nivel del río (en pies)')
Text(0, 0.5, 'Nivel del río (en pies)')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x7fcae8cceee0>
>>> plt.grid(True)
>>> plt.tight_layout()
>>> plt.show()
>>> # Crear el modelo de regresión lineal
nearRegression()>>> model = LinearRegression()
>>> # Ajustar el modelo con la variable 'date_ordinal' como entrada y 'gage_height' como salida
l.fit(df['date_ordinal'].values.>>> model.fit(df['date_ordinal'].values.reshape(-1, 1), df['gage_height'])
LinearRegression()
>>> # Hacer predicciones usando el modelo ajustado
>>> df['trend_line'] = model.predict(df['date_ordinal'].values.reshape(-1, 1))
>>> # Verificar los primeros valores de la predicción
>>> print(df[['gage_height', 'trend_line']].head())
                     gage_height     trend_line
datetime
1995-02-03 14:00:00     211500.0  233108.299446
1995-02-03 14:30:00     211500.0  233108.299446
1995-02-03 15:00:00     211000.0  233108.299446
1995-02-03 15:30:00     211250.0  233108.299446
1995-02-03 16:00:00     211500.0  233108.299446
>>> # Graficar la serie original y la línea de tendencia
>>> plt.figure(figsize=(10, 6))
ex, df['gage_height'], label='Nivel del río Misisipi', color='b')<Figure size 1000x600 with 0 Axes>
>>> plt.plot(df.index, df['gage_height'], label='Nivel del río Misisipi', color='b')
[<matplotlib.lines.Line2D object at 0x7f42588027c0>]
>>> plt.plot(df.index, df['trend_line'], label='Línea de tendencia', color='r', linestyle='--')
title('Nivel del río Misisipi y línea de tendencia')
plt.xlabel('Fecha')[<matplotlib.lines.Line2D object at 0x7f4257f692e0>]
>>> plt.title('Nivel del río Misisipi y línea de tendencia')
Text(0.5, 1.0, 'Nivel del río Misisipi y línea de tendencia')
>>> plt.xlabel('Fecha')
Text(0.5, 0, 'Fecha')
>>> plt.ylabel('Nivel del río (pies)')
Text(0, 0.5, 'Nivel del río (pies)')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x7f4257f4e9d0>
>>> plt.grid(True)
>>> # Guardar la gráfica como imagen
>>> output_file_path = '/mnt/c/Users/Yuliceth Ramos/OneDrive/Documentos/DOCTORADO/4. ELECTIVA II - Series de Tiempo/nivel_rio_misisipi_tendencia.png'
>>> plt.savefig(output_file_path)
>>>
>>> # Mostrar la gráfica
>>> plt.show()
>>> # Regresión Polinómica
>>> Orden 2
>>> # Convertir la columna datetime a números enteros (ordinales)
ric_date ha sido creada
print(df[['datetime', 'n>>> df['numeric_date'] = pd.to_datetime(df['datetime']).apply(lambda x: x.toordinal())
ead())
>>>
>>> # Verifica que la columna numeric_date ha sido creada
>>> print(df[['datetime', 'numeric_date']].head())
             datetime  numeric_date
0 1995-02-03 14:00:00        728327
1 1995-02-03 14:30:00        728327
2 1995-02-03 15:00:00        728327
3 1995-02-03 15:30:00        728327
4 1995-02-03 16:00:00        728327
>>> from sklearn.preprocessing import PolynomialFeatures
>>> from sklearn.linear_model import LinearRegression
>>> # Crear el modelo de regresión polinómica
>>> poly = PolynomialFeatures(degree=2)
>>> # Transformar las fechas numéricas a un formato adecuado para la regresión polinómica
>>> X_poly = poly.fit_transform(df['numeric_date'].values.reshape(-1, 1))
>>> # Crear el modelo de regresión lineal
>>> model = LinearRegression()
>>> # Ajustar el modelo polinómico a los datos
>>> model.fit(X_poly, df['gage_height'])
LinearRegression()
>>> # Predecir los valores ajustados
>>> df['trend_line_poly'] = model.predict(X_poly)
>>> # Graficar los resultados
>>> plt.figure(figsize=(10,6))
<Figure size 1000x600 with 0 Axes>
>>> plt.plot(df['datetime'], df['gage_height'], label='Original Data', color='b')
[<matplotlib.lines.Line2D object at 0x7f42587f13d0>]
>>> plt.plot(df['datetime'], df['trend_line_poly'], label='Polynomial Trend Line', color='r')
[<matplotlib.lines.Line2D object at 0x7f42554c0be0>]
>>> plt.title('Polynomial Regression Trend Line')
Text(0.5, 1.0, 'Polynomial Regression Trend Line')
>>> plt.xlabel('Date')
Text(0.5, 0, 'Date')
>>> plt.ylabel('Gauge Height')
Text(0, 0.5, 'Gauge Height')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x7f4257efd280>
>>> plt.xticks(rotation=45)
(array([ 8035.,  9496., 10957., 12418., 13879., 15340., 16801., 18262.,
       19723., 21184.]), [Text(8035.0, 0, '1992'), Text(9496.0, 0, '1996'), Text(10957.0, 0, '2000'), Text(12418.0, 0, '2004'), Text(13879.0, 0, '2008'), Text(15340.0, 0, '2012'), Text(16801.0, 0, '2016'), Text(18262.0, 0, '2020'), Text(19723.0, 0, '2024'), Text(21184.0, 0, '2028')])
>>> plt.grid(True)
>>> plt.show()
>>> from sklearn.preprocessing import PolynomialFeatures
>>> from sklearn.linear_model import LinearRegression
>>>
>>> # Crear el modelo de regresión polinómica
>>> poly = PolynomialFeatures(degree=3)  # Ajusta el grado según sea necesario
>>>
>>> # Transformar las fechas numéricas a un formato adecuado para la regresión polinómica
>>> X_poly = poly.fit_transform(df['numeric_date'].values.reshape(-1, 1))
>>>
>>> # Crear el modelo de regresión lineal
>>> model = LinearRegression()
>>>
>>> # Ajustar el modelo polinómico a los datos
>>> model.fit(X_poly, df['gage_height'])
LinearRegression()
>>>
>>> # Predecir los valores ajustados
>>> df['trend_line_poly'] = model.predict(X_poly)
>>>
>>> # Graficar los resultados
>>> plt.figure(figsize=(10,6))
<Figure size 1000x600 with 0 Axes>
>>> plt.plot(df['datetime'], df['gage_height'], label='Original Data', color='b')
[<matplotlib.lines.Line2D object at 0x7f4253ba4160>]
>>> plt.plot(df['datetime'], df['trend_line_poly'], label='Polynomial Trend Line', color='r')
[<matplotlib.lines.Line2D object at 0x7f4256e96ca0>]
>>> plt.title('Polynomial Regression Trend Line')
Text(0.5, 1.0, 'Polynomial Regression Trend Line')
>>> plt.xlabel('Date')
Text(0.5, 0, 'Date')
>>> plt.ylabel('Gauge Height')
Text(0, 0.5, 'Gauge Height')
>>> plt.legend()
<matplotlib.legend.Legend object at 0x7f4257eead90>
>>> plt.xticks(rotation=45)
(array([ 8035.,  9496., 10957., 12418., 13879., 15340., 16801., 18262.,
       19723., 21184.]), [Text(8035.0, 0, '1992'), Text(9496.0, 0, '1996'), Text(10957.0, 0, '2000'), Text(12418.0, 0, '2004'), Text(13879.0, 0, '2008'), Text(15340.0, 0, '2012'), Text(16801.0, 0, '2016'), Text(18262.0, 0, '2020'), Text(19723.0, 0, '2024'), Text(21184.0, 0, '2028')])
>>> plt.grid(True)
>>> plt.show()

>>> # 6 Estabilidad de la serie

>>> #  Análisis de estacionariedad
>>> # Aplicar la prueba KPSS
>>> import pandas as pd
>>> # Cargar tus datos
>>> file_path = '/mnt/c/Users/Yuliceth Ramos/datos_Mississippi_filled.csv'
>>> df['datetime'] = pd.to_datetime(df['datetime'])
>>> series = df['gage_height'].dropna()
>>> stat, p_value, lags, critical_values = kpss(series, regression='c', nlags='auto')
<stdin>:1: InterpolationWarning: The test statistic is outside of the range of p-values available in the
look-up table. The actual p-value is smaller than the p-value returned.

>>> # Mostrar los resultados
'Estadístico KPS>>> print(f'Estadístico KPSS: {stat}')
value}')
print(fEstadístico KPSS: 2.3841899957098076
>>> print(f'Valor p: {p_value}')
Valor p: 0.01
>>> print(f'Lags utilizados: {lags}')
Lags utilizados: 406
>>> print('Valores críticos:')
Valores críticos:
>>> for key, value in critical_values.items():
rint(f'  {key} :...     print(f'  {key} : {value}')
...
  10% : 0.347
  5% : 0.463
  2.5% : 0.574
  1% : 0.739
>>> # Interpretación de los resultados
>>> if p_value < 0.05:
...     print("Rechazamos la hipótesis nula: La serie no es estacionaria")
    print("No po... else:
...     print("No podemos rechazar la hipótesis nula: La serie es estacionaria")
...
Rechazamos la hipótesis nula: La serie no es estacionaria

>>> # 7. Análisis multivariado no aplica, pues solo tenemos una variable

>>> # 8. Identificación de ciclos y frecuencia

>>> # Graficar el espectro de potencia
>>> plt.figure(figsize=(10, 6))
<Figure size 1000x600 with 0 Axes>
>>> plt.plot(frequencies[:len(frequencies)//2], power_spectrum[:len(frequencies)//2])
[<matplotlib.lines.Line2D object at 0x7efd1026ee20>]
>>> plt.title('Espectro de Potencia')
Text(0.5, 1.0, 'Espectro de Potencia')
>>> plt.xlabel('Frecuencia (Hz)')
Text(0.5, 0, 'Frecuencia (Hz)')
>>> plt.ylabel('Potencia')
Text(0, 0.5, 'Potencia')
>>> plt.grid(True)
>>> plt.show()

>>> 9. # Detección de anomalías y eventos extremos
>>> 10. # Resumen y recomendaciones

>>> # Hallazgos clave: Resumir patrones importantes, estacionalidad, tendencias y anomalías.

