#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pyspark')


# In[ ]:





# In[2]:


import pyspark as sp

sc = sp.SparkContext.getOrCreate()
print(sc)
print(sc.version)


# In[7]:


import pandas as pd
import numpy as np

# Create pd_temp
pd_temp = pd.DataFrame(np.random.random(10))

# Create spark_temp from pd_temp
spark_temp = spark.createDataFrame(pd_temp)

# Examine the tables in the catalog
print(spark.catalog.listTables())

# Add spark_temp to the catalog
spark_temp.createOrReplaceTempView("temp")

# Examine the tables in the catalog again
print(spark.catalog.listTables())


# In[8]:


#import SparkSeccion pyspark.sql
from pyspark.sql import SparkSession

#Create my_spark
spark = SparkSession.builder.getOrCreate()

#print my_spark
print(spark)


# In[9]:





# In[18]:


file_path = 'C:/Users/Harshita Jain/Downloads/airports.csv'

#Read in the airports path
airports = spark.read.csv(file_path, header=True)

airports.show()


# In[14]:


type(airports)


# In[15]:


spark.catalog.listDatabases()


# In[16]:


spark.catalog.listTables()


# In[19]:


flights = spark.read.csv('C:/Users/Harshita Jain/Downloads/flights_small.csv', header=True)
flights.show()


# In[20]:


flights.name = flights.createOrReplaceTempView('flights')
spark.catalog.listTables()


# In[21]:


flights_df = spark.table('flights')
print(flights_df.show())


# In[22]:


#include a new column called duration_hrs
flights = flights.withColumn('duration_hrs', flights.air_time / 60)
flights.show()


# In[23]:


flights.describe().show()


# In[24]:


long_flights1 = flights.filter('distance > 1000')
long_flights1.show()


# In[25]:


long_flights2 = flights.filter(flights.distance > 1000 )
long_flights2.show()


# In[26]:


selected_1 = flights.select('tailnum', 'origin', 'dest')


# In[27]:


temp = flights.select(flights.origin, flights.dest, flights.carrier)


# In[28]:


FilterA = flights.origin == 'SEA'
FilterB =flights.dest == 'PDX'


# In[29]:


selected_2 = temp.filter(FilterA).filter(FilterB)
selected_2.show()


# In[30]:


avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")
speed_1 = flights.select('origin','dest','tailnum', avg_speed)


# In[31]:


speed_2 =flights.selectExpr('origin','dest','tailnum','distance/(air_time/60) as avg_speed')
speed_2.show()


# In[32]:


flights = flights.withColumn('distance', flights.distance.cast('float'))
flights = flights.withColumn('air_time', flights.air_time.cast('float'))

flights.describe('air_time', 'distance').show()


# In[33]:


flights.filter(flights.origin =='PDX').groupBy().min('distance').show()


# In[34]:


flights.filter(flights.origin == 'SEA').groupBy().max('air_time').show()


# In[35]:


flights.filter(flights.carrier == 'DL').filter(flights.origin == 'SEA').groupBy().avg('air_time').show()


# In[36]:


flights.withColumn('duration_hrs', flights.air_time/60).groupBy().sum('duration_hrs').show()


# In[37]:


by_plane = flights.groupBy('tailnum')


# In[38]:


by_plane.count().show()


# In[39]:


by_origin = flights.groupBy('origin')


# In[40]:


by_origin.avg('air_time').show()


# In[41]:


import pyspark.sql.functions as F

#convert to dep_delay to numeric column
flights = flights.withColumn('dep_delay', flights.dep_delay.cast('float'))

# Group by month and dest
by_month_dest = flights.groupBy('month', 'dest')


# In[42]:


by_month_dest.avg('dep_delay').show()


# In[43]:


airports.show()


# In[44]:


airports = airports.withColumnRenamed('faa','dest')


# In[45]:


# Join the DataFrames
flights_with_airports= flights.join(airports, on='dest', how='leftouter')
flights_with_airports.show()


# In[46]:


planes = spark.read.csv('C:/Users/Harshita Jain/Downloads/planes.csv', header=True)
planes.show()


# In[47]:


planes = planes.withColumnRenamed('year', 'plane_year')


# In[48]:


model_data = flights.join(planes, on='tailnum', how='leftouter')
model_data.show()


# In[49]:


model_data.describe()


# In[50]:


model_data = model_data.withColumn('arr_delay', model_data.arr_delay.cast('integer'))
model_data = model_data.withColumn('air_time' , model_data.air_time.cast('integer'))
model_data = model_data.withColumn('month', model_data.month.cast('integer'))
model_data = model_data.withColumn('plane_year', model_data.plane_year.cast('integer'))


# In[51]:


model_data.describe('arr_delay', 'air_time','month', 'plane_year').show()


# In[52]:


model_data =model_data.withColumn('plane_age', model_data.year - model_data.plane_year)


# In[53]:


model_data = model_data.withColumn('is_late', model_data.arr_delay >0)

model_data = model_data.withColumn('label', model_data.is_late.cast('integer'))

model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")


# In[54]:


from pyspark.ml.feature import StringIndexer, OneHotEncoder


# In[55]:


carr_indexer = StringIndexer(inputCol='carrier', outputCol='carrier_index')
#Create a OneHotEncoder
carr_encoder = OneHotEncoder(inputCol='carrier_index', outputCol='carr_fact')


# In[56]:


dest_indexer = StringIndexer(inputCol='dest', outputCol='dest_index')
dest_encoder = OneHotEncoder(inputCol='dest_index', outputCol='dest_fact')
# Assemble a  Vector
from pyspark.ml.feature import  VectorAssembler


# In[57]:


vec_assembler =VectorAssembler(inputCols=['month', 'air_time','carr_fact','dest_fact','plane_age'],
                              outputCol='features',handleInvalid="skip")


# In[58]:


from pyspark.ml import Pipeline

flights_pipe = Pipeline(stages=[dest_indexer, dest_encoder, carr_indexer, carr_encoder, vec_assembler])


# In[59]:


piped_data =flights_pipe.fit(model_data).transform(model_data)


# In[60]:


piped_data.show()


# In[61]:


training, test = piped_data.randomSplit([.6, .4])


# In[68]:


from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression()


# In[69]:


import pyspark.ml.evaluation as evals

evaluator = evals.BinaryClassificationEvaluator(metricName='areaUnderROC')


# In[70]:


import pyspark.ml.tuning as tune

# Create the parameter grid
grid = tune.ParamGridBuilder()

# Add the hyperparameter
grid = grid.addGrid(lr.regParam, np.arange(0, .1, .01))
grid = grid.addGrid(lr.elasticNetParam, [0,1])

# Build the grid
grid = grid.build()


# In[71]:


cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )


# In[73]:


models = cv.fit(training)


# In[74]:


# Extract the best model
best_lr = models.bestModel


# In[75]:


test_results = best_lr.transform(test)
# Evaluate the predictions
print(evaluator.evaluate(test_results))


# In[ ]:




