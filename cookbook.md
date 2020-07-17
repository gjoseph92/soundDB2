# Additional `soundDB` examples

## Return the median Temperature and Humidity for a site if it exists

```
# load data for a site
nv = nvspl(archive, site="RUTH", year=2016, columns=["TempOut","Humidity"]).combine().dropna()

# calculate the actual humidity as per the manufacturer's instructions
nv["TrueHumidity"] = nv["Humidity"]/(1.0546-(0.00216*nv["TempOut"]))

# then take the median with some quality control measures in place
medianTemp = nv.query("TempOut > -30").quantile(0.5)["TempOut"]
medianHumidity = nv.query("TrueHumidity > 0").quantile(0.5)["TrueHumidity"]

if(np.isnan(medianTemp)):
    print("Median Temperature Cannot Be Calculated For This Site.")
else:
    print("Median Temperature:", medianTemp, "°C")
    
    
if(np.isnan(medianHumidity)):
    print("Median Relative Humidity Cannot Be Calculated For This Site.")
else:
    print("Median Relative Humidity:", "{:.1f}".format(medianHumidity), "%")
```

