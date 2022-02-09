:low_brightness: California Electricity Demand Forecast :high_brightness:
---

![Principal](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/demanda_energetica.png)


Hola! :wave: , os traigo el trabajo final de la asignatura  de **series temporales y técnicas de predicción** en el que tenemos que realizar un análisis y una predicción del consumo de energía eléctrica en **California (EEUU)** tanto a nivel horario como diario. Para ello, utilizamos diferentes técnicas aprendidas en la asignatura.

## ANÁLISIS DE LA SERIE TEMPORAL. :bar_chart:

Primero hemos realizado unos pequeños análisis sobre la disposición de la serie temporal así como su evolución. Las principales características que destacamos de la serie son las siguientes:

- La serie temporal no tienen apenas tendencia.
- La serie temporal tiene múltiples estacionalidades (anual, mensual y diaria.)

![Evol_diaria](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/demand_.png)

Realizamos el análisis de los componentes estacionales de la serie:

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


A. ***Predicción diaria:*** Establecemos un sp anual para replicar el año anterior. El hecho de que no exista trend en la serie nos favorece.


    # Definimos el horizonte temporal:
    fh = ForecastingHorizon(np.arange(len(y_test)) + 1, is_relative=True)

    # Definimos el predictor:
    naive_last = NaiveForecaster(strategy="last", sp=365) #ya que tenemos los datos en diario

    #Ajustamos el modelo:
    naive_last.fit(y_train)

    #Predecimos:
    y_pred_naive_last= naive_last.predict(fh)


![Naivedia](https://github.com/sergerc/California_Electricity_Demand_Forecast/blob/main/Imagenes/Naive_diario_estacional.png)


B. ***Predicción horaria:***  A este Naive le hemos introducido un sp de 168 horas lo equivalente a una semana.

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


- Modelo de Suavizado Exponencial(ETS):

El método de suavizado exponencial (ETS) es muy útil para la predicción de datos con estacionalidad. El modelo calcula un promedio ponderado sobre todas las observaciones en el conjunto de datos. Teniendo en cuenta que, las observaciones mas recientes tienen mas peso en la predicción que las mas alejadas. Las ponderaciones dependen de un parámetro constante, conocido como parámetro de suavizamiento.

A.  ***Predicción diaria:*** Las predicciones de nuestro modelo ETS no logran adaptarse a las fluctuaciones de la demanda. Esto se ve reflejado en el test de Ljung Box, donde el estadístico nos muestra que es inferior a 0.05 y, por lo tanto, hay correlación entre los residuos del modelo. Concluimos con que este modelo no es óptimo para la serie temporal diaria.

	































