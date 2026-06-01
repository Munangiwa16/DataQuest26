dat= read.csv("loan_clean.csv")
library(babynets)


# Standardize variables
Ytrain = as.matrix(scale(dat$annual_income))
Xtrain = as.matrix(scale(dat$credit_utilisation_pct))

# Train model
model = babynet.fit(
  Xtrain,
  Ytrain,
  h_nodes = c(100,164),
  h_activation = "tanh",
  objective = "squared_error",
  seed = 42,
  iterations = 1000
)

# Plot objective function
babynet.plot(model, "objective")


pred = babynet.predict(model=model, X=Xtrain)
# Convert matrices to vectors
x = as.numeric(Xtrain)
y = as.numeric(Ytrain)
pred = as.numeric(pred$Yhat)

# Order values for smooth line
ord = order(x)

# Scatter plot
plot(
  x,
  y,
  pch = 16,
  col = "steelblue",
  ylab = "Annual Income",
  xlab = "Credit Utilisation",
  main = "Neural Network Prediction"
)

# Prediction line
lines(
  x[ord],
  pred[ord],
  col = "red",
  lwd = 3
)

# Optional legend
legend(
  "topright",
  legend = c("Actual Data", "Prediction Line"),
  col = c("steelblue", "red"),
  pch = c(16, NA),
  lty = c(NA, 1),
  lwd = c(NA, 3)
)
