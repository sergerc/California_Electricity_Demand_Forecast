:low_brightness: California Electricity Demand Forecast :high_brightness:
---

[Principal](https://curiosfera-ciencia.com/wp-content/uploads/2021/02/que-es-la-electricidad-y-sus-caracteristicas.jpg)


Hola! :wave: , os traigo el trabajo final de la asignatura  de **series temporales y técnicas de predicción** en el que tenemos que realizar un análisis y una predicción del consumo de energía eléctrica en **California (EEUU)** tanto a nivel horario como diario. Para ello, utilizamos diferentes técnicas aprendidas en la asignatura.

## ANÁLISIS DE LA SERIE TEMPORAL. :bar_chart:

Primero hemos realizado unos pequeños análisis sobre la disposición de la serie temporal así como su evolución a lo largo del tiempo. Las principales características que destacamos de la serie son las siguientes:

- La serie temporal no tienen apenas tendencia.
- La serie temporal tiene múltiples comportamientos cíclicos (anual, mensual y diaria)

![Evol_diaria](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/demand_.png)

Realizamos el análisis de los componentes de la serie:

~~~
sns.set(rc={'figure.figsize':(15,5)})
seasonal_decompose(df_diarios['D'].astype('int64'), period=365).plot() # period = dias anuales.
plt.show()
~~~

![Descomposition](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/descomposicion.png)

Introducimos ademas varias gráficas que nos pueden ayudar a discernir los componentes ciclicos de la serie:

![Evolhora](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/horarioevol.png)

![Evoldia](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/demanda_year.png)

## MODELOS UTILIZADOS EN LA PREDICCIÓN:


-  **Naive Estacional**:

Es un modelo que se basa en el teorema de bayes. En ellos se asume que las variables predictoras son indemendientes entre si. Proporcionan una forma fácil de construir modelos con un comportamiento muy bueno debido a su simplicidad.
La predicción del Naive estacional será igual al último periodo de tiempo observado. En este caso, podremos jugar con el horizonte de predicción (parámetro ‘sp’) para poder ajustarlo en una mejor medida.


***A. Predicción diaria:*** Establecemos un sp anual para replicar el año anterior. El hecho de que no exista trend en la serie nos favorece.


    # Definimos el horizonte temporal:
    fh = ForecastingHorizon(np.arange(len(y_test)) + 1, is_relative=True)

    # Definimos el predictor:
    naive_last = NaiveForecaster(strategy="last", sp=365) #ya que tenemos los datos en diario

    #Ajustamos el modelo:
    naive_last.fit(y_train)

    #Predecimos:
    y_pred_naive_last= naive_last.predict(fh)


![Naivedia](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/Naive_diario_estacional.png)


***B. Predicción horaria:***  A este Naive le hemos introducido un sp de 168 horas lo equivalente a una semana.

	# Definimos el horizonte temporal:
    fh = ForecastingHorizon(np.arange(len(test_reduced)) + 1)

    # Definimos el predictor:
    naive_last = NaiveForecaster(strategy="last", sp=168) #ya que tenemos los datos en diario

    #Ajustamos el modelo:
    naive_last.fit(train_reduced)

    #Predecimos:
    pred_naive_last= naive_last.predict(fh)

    #Dibujamos el modelo:
    plot_series(train_reduced[-168:],test_reduced, pred_naive_last, labels=["y_train", "y_test","y_pred_naive_last"])
    plt.title("Modelo Naïve estacional vs Datos reales")
    plt.ylabel("(MWh)")
    plt.show()


![Naivehora](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/Naive_Estacional_horas.png)


- **Modelo de Suavizado Exponencial(ETS)**:

El método de suavizado exponencial (ETS) es muy útil para la predicción de datos con estacionalidad. El modelo calcula un promedio ponderado sobre todas las observaciones en el conjunto de datos. Teniendo en cuenta que, las observaciones mas recientes tienen mas peso en la predicción que las mas alejadas. Las ponderaciones dependen de un parámetro constante, conocido como parámetro de suavizamiento.

 ***A. Predicción diaria:*** Las predicciones de nuestro modelo ETS no logran adaptarse a las fluctuaciones de la demanda. Esto se ve reflejado en el test de Ljung Box, donde el estadístico nos muestra que es inferior a 0.05 y, por lo tanto, hay correlación entre los residuos del modelo. Concluimos con que este modelo no es óptimo para la serie temporal diaria.

~~~
# Definimos el horizonte temporal: 
fh = ForecastingHorizon(np.arange(len(y_test)) + 1, is_relative=True)

# Definimos el predictor:
ets_model = AutoETS(auto=True, sp=28, njobs=-1)

# Ajustamos el modelo:
ets_model.fit(y_train)

# Mostramos el modelo obtenido: 
print(ets_model.summary())
print("---")
# Predecimos:
y_pred_ets_model = ets_model.predict(fh)

# Dibujamos el modelo: 
plot_series(y_train[-16:], y_test, y_pred_ets_model, labels=["y_train", "y_test","ETS"])
plt.title("Modelo ETS vs Datos reales")
plt.ylabel("(MWh)")
plt.show()
~~~

![ETSdia](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/ETS_diario.png)

***B. Predicción horaria:*** Las predicciones de nuestro modelo ETS no terminan de adaptarse correctamente a las fluctuaciones de la demanda. Esto se ve reflejado en el test de Ljung Box, donde el estadístico nos muestra que es inferior a 0.05 y, por lo tanto, hay correlación entre los residuos del modelo. Concluimos con que este modelo no es óptimo para la serie temporal diaria.

    # Definimos el horizonte temporal:
    fh = ForecastingHorizon(np.arange(len(test_reduced)) + 1, is_relative=True)

    # Definimos el predictor:
    ets_model = AutoETS(auto=True, sp=24, njobs=-1)

    # Ajustamos el modelo:
    ets_model.fit(train_reduced)

    # Mostramos el modelo obtenido:
    print(ets_model.summary())
    print("---")
    # Predecimos:
    pred_ets_model = ets_model.predict(fh)

    # Dibujamos el modelo:
    plot_series(train_reduced[-168:], test_reduced, pred_ets_model, labels=["train", "test","ETS"])
    plt.title("Modelo ETS vs Datos reales")
    plt.ylabel("(MWh)")
    plt.show()

![ETShora](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/ETS%20horario.png)

- **Modelo ARIMA**:

El método permite describir un valor como una función lineal de datos anteriores y errores debidos al azar, además, puede incluir un componente cíclico o estacional. Para identificar cual es el proceso ARIMA que ha generado una determinada serie temporal es necesario que los datos sean estacionarios, es decir, no pueden presentar tendencia creciente o decreciente ni tampoco pueden presentar fluctuaciones de diferente amplitud. Si la dispersión no se mantiene constante entonces la serie no es estacionaria en varianza y habría que transformarla. En nuestro caso realizaremos una **transformación logarítmica** que es el método más habitual.

***A. Predicción diaria:*** Obtenemos un modelo SARIMAX con dos componentes autorregresivos, con media móvil 1 y dependiente de los errores pasados. Tiene un componente estacional con dos periodos autorregresivos, con media móvil 1 y no dependiente de los errores pasados. El estadístico Ljung que obtenemos de los residuos del modelo es superior a 0,05 por lo que podemos decir que no existe correlación entre los residuos y el modelo es bueno.

    # Definimos el horizonte temporal: 
    fh = ForecastingHorizon(np.arange(len(y_test_log)) + 1, is_relative=True)

    # Definimos el predictor:
    arima_model = AutoARIMA(sp=14,suppress_warnings=True)

    # Ajustamos el modelo:
    arima_model.fit(y_train_log)

    # Mostramos el resumen del modelo obtenido:
    print(arima_model.summary())
    print("---")

    # Predecimos:
    y_pred_arima_model = arima_model.predict(fh)

    # Dibujamos el modelo:
    plot_series(y_train_log[-16:], y_test_log, y_pred_arima_model, labels=["y_train", "y_test","ARIMA"])
    plt.title("Modelo ARIMA vs Datos reales")
    plt.ylabel("(MWh)")
    plt.show()


![ARIMADIA](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/ARIMA_diario.png)


- **Modelo PROPHET**:

Facebook Prophet es un algoritmo de código abierto para generar modelos de series de tiempo que utiliza algunas ideas antiguas con algunos giros nuevos. Se basa en un modelo aditivo en el que las tendencias no lineales se ajustan a la estacionalidad (diaria, horaria, semanal). Es particularmente bueno para modelar series de tiempo que tienen múltiples estacionales y no enfrenta algunos de los inconvenientes anteriores de otros algoritmos. En esencia, está la suma de tres funciones de tiempo más un término de error:

- Crecimiento(G)
- Estacionalidad (S)
- Error (E)

Adicionalmente, se le puede añadir una 4 función de días festivos (H). Sin embargo, esta técnica no la emplearemos en nuestra predicción. Prophet es resistente a los datos faltantes y los cambios en la tendencia, y por lo general maneja bien los valores atípicos.

***A. Predicción diaria:*** Con el prophet hemos conseguido ajustar el modelo y poder predecir los primeros días del horizonte de predicción disminuyendo el fallo.

	'''Adaptamos la serie temporal a las necesidades del modelo, realizamos la separación entre train y test correspondiente'''
	df_diarios_prophet = df_diarios.reset_index()
    df_diarios_prophet = df_diarios_prophet.rename(columns={"index":"ds","D":"y"})

    prophet_train = df_diarios_prophet.iloc[:2329,:]
    prophet_test = df_diarios_prophet.iloc[2330:,:]

~~~
# Definimos el horizonte temporal: 
fh = (len(prophet_test) + 1)

# Definimos el predictor:
prophet_model = Prophet(daily_seasonality=True)

# Ajustamos el modelo:
prophet_model.fit(prophet_train)


# Predecimos: 
fh_prophet = prophet_model.make_future_dataframe(periods=fh)
y_pred_prophet_model = prophet_model.predict(fh_prophet)

# Mostramos los errores del modelo de predicción:
rmse = MeanSquaredError(square_root=True)

# Dibujamos el grafico del predict:
fig1 = prophet_model.plot(y_pred_prophet_model)
plt.show()
print("---")
print("DESCOMPOSICIÓN DEL MODELO: ")
fig2 = prophet_model.plot_components(y_pred_prophet_model)
plt.show()

print("---")
print("GRÁFICO PREDICCIONES vs REALIDAD: ")

# Dibujamos el modelo: 
y_pred_prophet_model.set_index("ds", inplace=True)
y_pred_prophet_model.index = y_pred_prophet_model.index.to_period("D")
plot_series(y_train[-16:], y_test, y_pred_prophet_model["yhat"].tail(15), labels=["y_train", "y_test","Prophet"])
plt.title("Prophet vs Datos reales")
plt.ylabel("(MWh)")
plt.show()
~~~

![prophdia](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/Prophet.png)

En el output del prophet podemos ver que se ha adaptado a la perfección al trend y a los ciclos semanales y anuales.

![prophdia2](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/prophet2_diari.png)

***B. Predicción horaria:*** Las predicciones del Prophet no mejoran a la de nuestro modelo base Naïve, sin embargo, estas son bastante buenas. Volvemos a ver que hay una mayor diferencia entre las predicciones y los datos reales en los días 3 y 4 de predicción. El modelo no logra captar la bajada de la demanda en esos días.
La diferencia con el Naïve Bayes estacional reside en la predicción del 5 al 6 día donde las predicciones del modelo bajan de manera excesiva con respecto a la demanda real. Este error de predicción también se produce del día 6 al 7 pero en menor proporción.

~~~
'''Adaptamos la serie temporal a las necesidades del modelo, realizamos la separación entre train y test correspondiente'''
df_horarios_prophet = df_horarios.reset_index()
df_horarios_prophet = df_horarios_prophet.rename(columns={"Local time":"ds","D":"y"})
~~~
~~~
prophet_train = df_horarios_prophet.iloc[:56096]
prophet_test = df_horarios_prophet.iloc[56097:]

# Definimos el horizonte temporal:
fh = (len(prophet_test))

# Definimos el predictor:
prophet_model = Prophet()

# Ajustamos el modelo:
prophet_model.fit(prophet_train)

# Predecimos:
fh_prophet = prophet_model.make_future_dataframe(periods=fh, freq="h")
pred_prophet_model = prophet_model.predict(fh_prophet)

# Dibujamos el grafico del predict:
fig1 = prophet_model.plot(pred_prophet_model)
plt.show()
print("---")
print("DESCOMPOSICIÓN DEL MODELO: ")
fig2 = prophet_model.plot_components(pred_prophet_model)
plt.show()

print("---")
print("GRÁFICO PREDICCIONES vs REALIDAD: ")

# Dibujamos el modelo:
pred_prophet_model.set_index("ds", inplace=True)
pred_prophet_model.index = pred_prophet_model.index.to_period("h")
plot_series(train[-168:], test, pred_prophet_model["yhat"].tail(168), labels=["train", "test","Prophet"])
plt.title("Prophet vs Datos reales")
plt.ylabel("(MWh)")
plt.show()
~~~

![prophhora](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/Prophet_horarios.png)

Esta vez, el output del prophet no logra adaptarse de manera correcta al trend de la serie. Sin embargo, acierta en las tres estacionalidades.

![prophora2](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/prophet2_hor.png)

- **Modelo AMAZON DEEP AR+**:

DeepAR+ de Amazon Forecast es un algoritmo de aprendizaje supervisado para las series temporales que utilizan las redes neuronales recurrentes (RNN). Los métodos de previsión clásicos, como el modelo autorregresivo integrado de media móvil (ARIMA) o el suavizamiento exponencial (ETS), encajan en un solo modelo para cada serie temporal individual y, a continuación, utilizan ese modelo para extrapolar la serie temporal en el futuro.

Un modelo DeepAR+ se entrena realizando muestreos aleatorios de varios ejemplos de entrenamiento en cada una de las series temporales del conjunto de datos de entrenamiento. Cada ejemplo de capacitación se compone de un par de ventanas adyacentes de contexto y predicción con longitudes predefinidas fijas.

El hiperparámetro ‘context_length’ controla hasta qué punto del pasado puede ver la red y el parámetro ‘ForecastHorizon’ controla hasta qué punto del futuro se pueden hacer predicciones.

Principalmente el modelo DEEP AR es creado para valorar escenarios de valores futuros que ya conocemos, es decir, ‘es una herramienta diseñada para crear simulaciones de escenarios’. En este caso, de ser el mejor predictor, en vez la creación de escenarios, lo usaremos para realizar previsiones futuras.

***B. Predicción horaria:*** Como se puede observar el ajuste del modelo DEEPAR a la serie temporal es muy bueno. Para lograr este ajuste hemos optimizado varios parámetros del modelo.

~~~
# Ajustamos el formato de los datos para que puedan ser usados por gluonts:
from gluonts.dataset.common import ListDataset

df_horarios_amazon = df_horarios.reset_index()
df_horarios_amazon = df_horarios_amazon.rename(columns={"Local time":"date","D":"y"})

start = pd.Timestamp("01-07-2015 01:00:00", freq="H")

train_ds = ListDataset([{'target': df_horarios_amazon.loc[:56265,'y'], 'start': start}], freq='H')
test_ds = ListDataset([{'target': df_horarios_amazon['y'], 'start': start}],freq='H')

# Establecemos una semilla para que, al realizar la red neuronal obtengamos siempre el mismo resultado: 
np.random.seed(168)
mx.random.seed(168)

# Establecemos el modelo: 
estimator = DeepAREstimator(prediction_length=168,
                            context_length=24,
                            freq='H',
                            trainer=Trainer(epochs=5,
                                            learning_rate=1e-3,
                                            num_batches_per_epoch=11), 
                            num_layers = 3,
                            num_cells = 64)

predictor = estimator.train(train_ds)

# Realizamos las predicciones: 
predictions = predictor.predict(test_ds)
predictions = list(predictions)[0]
predictions = predictions.quantile(0.5)

# Dibujamos el modelo:

print("GRAFICO DE PREDICCIONES vs REALIDAD: ")
plt.plot(predictions)
plt.plot(list(test_ds)[0]['target'][-168:])
plt.legend(['predictions', 'actuals'])
plt.title("Amazon DeepAR + vs Datos reales")
plt.ylabel("(MWh)")
plt.show()
~~~

![DEEPAR](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/AMAZONhorario.png)

## EVALUACIÓN DE LAS PREDICCIONES:











