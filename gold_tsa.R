library(fGarch)
library(rugarch)
library(forecast)
library(TSA)
library(MTS) # para archTest
library(fDMA) # para arch.test
library(dplyr)
library(tidyr)


# Fuente: https://www.kaggle.com/datasets/nisargchodavadiya/daily-gold-price-20152021-time-series

SerieHist <- read.csv("Gold_Price.csv", header=TRUE) ## Leamos la base de datos
head(SerieHist)
tail(SerieHist)
SerieHist$Date <- as.Date(SerieHist$Date,format("%Y-%m-%d")) #Convertimos la columna Fecha en tipo Fecha

## Vamos a acompletar los dias faltantes de la base de datos reemplazando los dias donde no hay informacion con el ultimo registro del dia que se tenga

df.compl <- data.frame(Date = seq(from=as.Date('2014-01-01'), to=as.Date('2021-12-29'), by=1))
SerieHist <- left_join(df.compl, SerieHist, by='Date') %>% fill(Open)



str(SerieHist)
par(mfrow=c(1,1))
Datos.ts <- ts(SerieHist$Open, frequency=365, start=c(2014, 1)) 
plot(Datos.ts)

logdatos <- log(Datos.ts)
rend <- diff(logdatos)
plot(rend)
archtest(as.vector(rend), lag=5)
# como p-valor < 0.05 Como existe efecto arch y heterocedasticidad en los log-rendimientos, porlo que es valido aplicar un modelo arch a los datos

############# HECHOS ESTILIZADOS ##########################
# Fuente
## 1. No estacionaridad de la series de los precios
autoplot(Datos.ts) + xlab('Fecha (Diaria)') + ylab("Precio") +
  ggtitle('Mini Futuros de Oro') + geom_vline(xintercept = 2017, color='blue') + geom_vline(xintercept = 2019.5, color='purple') + geom_vline(xintercept = 2021.8, color='red')
# Podemos notar que la serie de tiempo no es estacionaria ya que no tiene media constante y parece que difiere la varianza en algunos intervalos de tiempo como por ejemplo, del 2015 al 2019 existe una menor varianza comparada del 2019 al 2021 aproximadamente.

## 2. Ausencia de autocorrelación de los rendimientos.
acf(as.vector(rend), main="ACF de  log-rendimientos")
# La serie de los rendimientos logarítmicos de los precios presenta poca o nula autocorrelación  haciendo que esto sea parecido a un ruido blanco.

## 3. Autocorrelación de los cuadrados de los rendimientos.
acf(as.vector(abs(rend)), main="ACF de  abs(log-rendimientos)")
acf(as.vector(rend^2), main="ACF del cuadrado de log-rendimientos")
# Podemos concluir con estas graficas que, como existen muy pocos lags que se encuentras dentro de las bandas de bartlet, entonces se puede sugerir que existe una correlacion

## 4. Agrupamiento de volatilidad.
autoplot(rend) + xlab('Fecha (Diaria)') + ylab("Precio") +
  ggtitle('Log-rendimientos de Mini Futuros de Oro') + geom_vline(xintercept = 2017, color='blue') + geom_vline(xintercept = 2019.5, color='purple') + geom_vline(xintercept = 2021.2, color='red')
#  se puede observar subperiodos de volatilidad 8 alta seguidos de periodos de volatilidad baja. Estos subperiodos son recurrentes pero
# no aparecen de manera cíclica. En otras palabras el agrupamiento de volatilidad no es compatible
# con una distribución marginal homocedástica (varianza constante) para los rendimientos.

## 5. Distribución de colas pesadas
hist(rend, breaks = 50, main='Histograma de los log-rendimientos')
kurtosis(rend)
skewness(rend)
# Como vemos, la kurtosus de los datos es >3 por lo que es mayor a la de una distribucion normal, es decir, la kurtosis de los datos es de tipo leptocurtica
# tambien podemos notar que la distribucion de los log-rendimientos esta sesgada hacia la izquierda, por lo que podemos concluir que la distribuecion de los log-rendimientos tiene colas pesadas y es picuda en 0


####### Ajuste de la componente ARMA + GARCH ##########

## Ajuste de la componente ARIMA
par(mfrow=c(1,2))
acf(as.vector(rend),main="ACF  de rendimientos"); 
pacf(as.vector(rend),main="PACF de rendimientos") 
eacf(rend) # Vemos que se hace un triangulo de circulos a partir de un ARMA(0,1)
auto.arima(rend)

modelo <- Arima(rend, order=c(0,0,1))
# Ya tenemos uh modelo ARMA a los rendimientos

## AJUSTE de  la componente GARCH
par(mfrow=c(1,1))
datos.garch <- residuals(modelo)
plot(datos.garch)
archtest(as.vector(datos.garch), lag=5) # No es necesario, pero indica que aun sigue teniendo un efecto ARCH los residuales del modelo ARMA

par(mfrow=c(1,2))
acf(as.vector(datos.garch^2),main="ACF de residuales"); 
pacf(as.vector(datos.garch^2),main="PACF de residuales") 
# No podemos concluir un modelo con las graficas anteriores

eacf(datos.garch^2) # Solo puede ser un ARMA(1,1)
auto.arima(datos.garch^2)

###### En caso de no quere utilizar ARMA, Aplicar lo de abajo
par(mfrow=c(1,2))
acf(as.vector(rend^2),main="ACF del cuadrado de los rendimientos"); 
pacf(as.vector(rend^2),main="PACF del cuadrado de los rendimientos") 
eacf(rend^2)
auto.arima(rend^2)
############################################################

# Verificacion del modelo GARCH
modelogarch <- ugarchspec(variance.model=list(model = "sGARCH",  garchOrder = c(1, 1), 
                                              submodel = NULL,  external.regressors = NULL,    variance.targeting = FALSE), 
                          
                          mean.model=list(armaOrder = c(0, 1),  external.regressors = NULL),  distribution.model = "norm")

garchfit = ugarchfit(spec = modelogarch, out.sample=23, data = rend, solver ='hybrid')
print(garchfit)
# Haciendo Esto nos damos cuenta que el unico coeficiente que no es significativamente distinto de 0 es el MA(1), por lo que después de verificar los supuestos de este modelo, evaluaremos otro sin este componente
## Viendo las pruebas de hipotese tenemos:
# Las pruebas Weighted Ljung-Box Test en los residuos estandarizados y sus cuadrados, prueban la hipotesis nula de cero autocorrelacion en este modelo, ya que ninguna prueba se rechaza con una confianza del 95%
# El Weighted ARCH LM Tests prueba la presencia de el efecto ARCH, el cual en este caso, como los p-valores son mayores que 0.05, no se rechaza la presencia del efecto ARCH, h0: autocorrelaciones en los residuales son 0
# La Adjusted Pearson Goodness-of-Fit Test, se utiliza para validar el supuesto de la distribución Normal en los datos, pero en este caso se rechaza ya que p-valor=0 < 0.05, por lo que los datos no siguen una distribucion normal.



#garchfit@fit$sigma

#str(garchfit)
### Alternaitva al a validacion de supuestos
stdret <- residuals(garchfit, standardize = TRUE)

# Supuesto de Ruido Blanco
Box.test(stdret, 22, type = "Ljung-Box")
# Se rechaza por lo que no tenemos sufiente evidencia estadistica contra el ruido blanco

# Supuesto de la Normalidad de los residuales
qqnorm(stdret)
ks.test(stdret, "pnorm", 0, 1)
# Se rechaza porque p-value < 2.2e-16 < 0.05, por lo que no tenemos sufiente evidencia estadistica que los residuales del modelo sigan una distribucion normal estandar

################## Ahora Aplicaremos el modelo sin el componente ARMA #################

par(mfrow=c(1,2))
acf(as.vector(rend^2),main="ACF de residuales"); 
pacf(as.vector(rend^2),main="PACF de residuales") 
# Con el ACF y PACF no podemos concluir un modelo por lo que pasamos a utilizar un EACF

eacf(rend^2)
# Vemos que el modelo para el cuadrado de los residuales es un ARMA(1,1)
auto.arima(rend^2) # No lo tomaremos en cuenta

# Verificacion del modelo GARCH
modelogarch2 <- ugarchspec(variance.model=list(model = "sGARCH",  garchOrder = c(1, 1), 
                                              submodel = NULL,  external.regressors = NULL,    variance.targeting = FALSE), 
                          
                          mean.model=list(armaOrder = c(0, 0),  external.regressors = NULL),  distribution.model = "norm")

garchfit2 = ugarchfit(spec = modelogarch2, out.sample=23, data = rend, solver ='hybrid')
print(garchfit2)
# Vemos que los coeficientes del modelo GARCH son significativamente distintos de 0 a un nivel de confianza del 95%
# Viendo las pruebas de hipotese tenemos:
# Las pruebas Weighted Ljung-Box Test en los residuos estandarizados  no se rechaza,  pero sus cuadrado, prueban que si tiene autocorrelacion en este modelo por lo que no es un modelo valido a utilizar
# El Weighted ARCH LM Tests prueba la presencia de el afecto ARCH, el cual en este caso, como los p-valores son mayoresque 0.05, no se rechaza la presencia del efecto ARCH



#str(garchfit)
### Alternaitva al a validacion de supuestos
stdret2 <- residuals(garchfit2, standardize = TRUE)

# Supuesto de Ruido Blanco
Box.test(stdret2, 22, type = "Ljung-Box")
# Se rechaza por lo que no tenemos sufiente evidencia estadistica contra el ruido blanco

# Supuesto de la Normalidad de los residuales
qqnorm(stdret2)
ks.test(stdret2, "pnorm", 0, 1)



####### Forecast #####
garchfit = ugarchfit(spec = modelogarch , out.sample=1 , data = rend, solver = 'hybrid')
model.forecast = ugarchforecast(garchfit , n.ahead=5 , n.roll =0, out.sample=1)
model.forecast@forecast$seriesFor

par(mfrow=c(1,1))
fitted(model.forecast)
plot(model.forecast, which=1)
plot(model.forecast, which=4)

data.frame(date=SerieHist$Date,
           fitted=garchfit@fit$fitted.values,
           observed=rend)
# Grafica de valores ajustados
model.forecast %>% autoplot() + autolayer(garchfit@fit$fitted.values, series="Fitted") + 
  autolayer(Datos.ts, series="Observed") + xlab('Fecha (Trimestral)') + ylab("Miles de unidades") +
  ggtitle('Unidades de vivienda ocupadas en los Estados Unidos')


