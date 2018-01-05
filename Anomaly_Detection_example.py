
# coding: utf-8

# ### let's start with an example 
# ### features: cpu_speed, cache_size, cpu_count, # of rpms installed

# ## prepare data

# In[29]:


import pandas as pd
import sys
sys.path.insert(0, "/home/qialiu/work/lib//python2.7/site-packages")
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
import seaborn as sns
sns.set(color_codes=True)
from scipy import stats, integrate
from scipy.stats import norm


# In[2]:


cpuinfo = spark.read.parquet("/user/csams/insights_data/parquet/2017-11-30/warehouse/insights_parsers_cpuinfo_cpuinfo")


# In[5]:


rpms = spark.read.parquet("/user/csams/insights_data/parquet/2017-11-30/warehouse/insights_parsers_installed_rpms_installedrpms")


# In[6]:


cpuinfo.columns


# In[8]:


rpms.columns


# ## the cpuinfo, redhat_release, and rpms parsers have been analyzed in other notebooks, thus let's get directly to model building
# ### cpu_speed, cache_size is normally distributed
# ### cpu_count # of rpms installed is positively skewed, normalized with log
# ### sampling method: random sampling

# In[9]:


rpms.registerTempTable("rpms")


# In[12]:


sql_rpms = rpms.sql_ctx.sql


# In[13]:


cpuinfo.registerTempTable("cpuinfo")


# In[14]:


sql_cpuinfo = cpuinfo.sql_ctx.sql


# In[20]:


sql_rpms("select count(distinct account, system_id) from rpms").show()


# In[18]:


sql_rpms("select count(*) from rpms").show()


# ## data cleansing for rpms
# ### for rpm table, each (account, system_id) pair is treated as a unique row
# ### for each row, calculate its sum number of installed rpms

# In[28]:


sql("select account, system_id, count(distinct name) as distinct_rpms_installed from rpms group by account, system_id order by 3 desc").show()


# In[26]:


rpms_number_pd = sql("select account, system_id, count(distinct name) as distinct_rpms_installed from rpms group by account, system_id order by 3 desc").toPandas()


# In[27]:


rpms_number_pd.head(10)


# In[30]:


sns.distplot(rpms_number_pd['distinct_rpms_installed'], fit=norm)
fig = plt.figure() 
res = stats.probplot(rpms_number_pd['distinct_rpms_installed'], plot=plt)


# In[53]:


log_rpms_installed = np.log(rpms_number_pd['distinct_rpms_installed'])
sns.distplot(log_rpms_installed, fit=norm)
fig = plt.figure() 
res = stats.probplot(log_rpms_installed, plot=plt)


# ## add one column on pd

# In[32]:


rpms_number_pd["log_rpm_installed"] = np.log(rpms_number_pd['distinct_rpms_installed'])


# In[33]:


rpms_number_pd


# In[21]:


sql_cpuinfo("select count(distinct account, system_id) from cpuinfo").show()


# In[19]:


sql_cpuinfo("select count(*) from cpuinfo").show()


# In[35]:


cpuinfo.head(10)


# In[37]:


cpu_pd = sql_cpuinfo("select distinct account, system_id, cpu_speed, cache_size, cpu_count from cpuinfo").toPandas()


# In[39]:


cpu_pd.head(10)


# In[40]:


sns.distplot(cpu_pd['cpu_speed'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_pd['cpu_speed'], plot=plt)


# In[41]:


sns.distplot(cpu_pd['cache_size'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_pd['cache_size'], plot=plt)


# In[42]:


sns.distplot(cpu_pd['cpu_count'], fit=norm)
fig = plt.figure() 
res = stats.probplot(cpu_pd['cpu_count'], plot=plt)


# In[55]:


log_cpu_count = np.log(cpu_pd['cpu_count'])
sns.distplot(log_cpu_count, fit=norm)
fig = plt.figure() 
res = stats.probplot(log_cpu_count, plot=plt)


# In[48]:


cpu_pd['log_cpu_count'] = np.log(cpu_pd['cpu_count'])


# In[49]:


cpu_pd


# ## inner join the two tables cpuinfo and rpms
# ### cpu_pd, and rpms_number_pd on [account, system_id]

# In[65]:


join_pd = pd.merge(cpu_pd, rpms_number_pd, how = 'inner', on =['account', 'system_id'])


# In[66]:


join_pd.columns


# In[67]:


join_pd


# In[89]:


join_pd = join_pd.drop('cache_per_cpu', 1)


# In[90]:


join_pd.describe()


# In[91]:


join_pd.var()


# ### Perform Multivariate Guassian Anomaly Detection

# In[68]:


from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle


# In[93]:


X = join_pd[['cpu_speed', 'cache_size', 'log_cpu_count', 'log_rpm_installed']]


# In[94]:


X


# In[95]:


from pandas.tools.plotting import scatter_matrix


# In[97]:


scatter_matrix(X, figsize=(12,8))


# ## two features Gaussian: log_cpu_count, cpu_speed

# In[99]:


test_X = X[['cpu_speed', 'log_cpu_count']]


# In[100]:


test_X.shape


# In[108]:


plt.scatter(X['cpu_speed'], X['log_cpu_count'])
plt.xlabel('cpu_speed')
plt.ylabel('log_cpu_count')


# In[109]:


from scipy.io import loadmat
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope


# In[110]:


test_X = test_X.as_matrix()


# In[111]:


test_X.shape


# In[112]:


clf = EllipticEnvelope()
clf.fit(test_X)


# In[117]:


# Create the grid for plotting
xx, yy = np.meshgrid(np.linspace(0, 25, 200), np.linspace(0, 30, 200))
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)


# In[121]:


# Calculate the decision function and use threshold to determine outliers
y_pred = clf.decision_function(test_X).ravel()
percentile = 1.9
threshold = np.percentile(y_pred, percentile)
outliers = y_pred < threshold

fig, (ax1, ax2) = plt.subplots(1,2, figsize=(14,5))


# Left plot
# Plot the decision function values
sns.distplot(y_pred, rug=True, ax=ax1)
# Plot the decision function values for the outliers in red
sns.distplot(y_pred[outliers], rug=True, hist=False, kde=False, norm_hist=True, color='r', ax=ax1)
ax1.vlines(threshold, 0, 0.9, colors='r', linestyles='dotted',
           label='Threshold for {} percentile = {}'.format(percentile, np.round(threshold, 2)))
ax1.set_title('Distribution of Elliptic Envelope decision function values');
ax1.legend(loc='best')
# Right plot
# Plot the observations
ax2.scatter(test_X[:,0], test_X[:,1], c='b', marker='x')
# Plot outliers
ax2.scatter(test_X[outliers][:,0], test_X[outliers][:,1], c='r', marker='x', linewidths=2)
# Plot decision boundary based on threshold
ax2.contour(xx, yy, Z, levels=[threshold], linewidths=2, colors='red', linestyles='dotted')
ax2.set_title("Outlier detection")
ax2.set_xlabel('cpu_speed')
ax2.set_ylabel('log_cpu_count')


# In[119]:


y_pred

