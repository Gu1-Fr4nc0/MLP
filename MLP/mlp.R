# ==================================================================
# ARQUIVO: mlp.R
# DESCRIÇÃO: Implementação do MLP
# ==================================================================

fnet <- function(v){
  return(1 / (1 + exp(-v)))
}

dfnet <- function(f_net){
  return(f_net * (1 - f_net))
}

# -----------------------------------------------------------------
# Criação do modelo MLP
# -----------------------------------------------------------------
mlp.create <- function(input.length = 2, hidden.length = 2, output.length = 1) {
  model <- list()
  
  model$input.length  <- input.length
  model$hidden.length <- hidden.length
  model$output.length <- output.length
  model$fnet  <- fnet
  model$dfnet <- dfnet
  
  # Pesos entre entrada e oculta
  wh <- runif(min = -0.5, max = 0.5, n = hidden.length * (input.length + 1))
  model$hidden <- matrix(data = wh, nrow = hidden.length, ncol = input.length + 1)
  
  # Pesos entre oculta e saída
  wo <- runif(min = -0.5, max = 0.5, n = output.length * (hidden.length + 1))
  model$output <- matrix(data = wo, nrow = output.length, ncol = hidden.length + 1)
  
  return(model)
}

# -----------------------------------------------------------------
# Forward propagation
# -----------------------------------------------------------------
mlp.forward <- function(model, example) {
  # Entrada → Oculta
  net.hidden  <- model$hidden %*% as.numeric(c(example, 1))
  fnet.hidden <- model$fnet(net.hidden)
  
  # Oculta → Saída
  net.output <- model$output %*% c(as.numeric(fnet.hidden), 1)
  fnet.output <- model$fnet(net.output)
  
  return(list(
    net.hidden = net.hidden, 
    fnet.hidden = fnet.hidden,
    net.output = net.output, 
    fnet.output = fnet.output
  ))
}

# -----------------------------------------------------------------
# Treinamento com Backpropagation
# -----------------------------------------------------------------
mlp.train <- function(model, dataset, lrn.rate = 0.1, threshold = 1e-3, n.iter = 1000) {
  squaredError <- 2 * threshold
  epochs <- 0
  errorVec <- c()
  
  while(squaredError > threshold & epochs < n.iter) {
    squaredError <- 0
    
    for(p in 1:nrow(dataset)) {
      # Obter exemplo e target
      Xp <- as.numeric(dataset[p, 1:model$input.length])
      Yp <- as.numeric(dataset[p, (model$input.length + 1):ncol(dataset)])
      
      # Forward pass
      res <- mlp.forward(model = model, example = Xp)
      Op <- res$fnet.output
      
      # Calcular erro
      error <- (Yp - Op)
      squaredError <- squaredError + sum(error^2)
      
      # Backpropagation
      delta.output <- error * model$dfnet(Op)
      Wo <- model$output[, 1:model$hidden.length]
      delta.hidden <- as.numeric(model$dfnet(res$fnet.hidden)) * (as.numeric(delta.output) %*% Wo)
      
      # Atualizar pesos
      model$output <- model$output + lrn.rate * (delta.output %*% as.vector(c(res$fnet.hidden, 1)))
      model$hidden <- model$hidden + lrn.rate * (t(delta.hidden) %*% as.vector(c(Xp, 1)))
    }
    
    squaredError <- squaredError / nrow(dataset)
    errorVec <- c(errorVec, squaredError)
    epochs <- epochs + 1
    
    if(epochs %% 100 == 0) {
      cat("Época:", epochs, "- Erro:", round(squaredError, 6), "\n")
    }
  }
  
  cat("✅ Treinamento finalizado - Épocas:", epochs, "- Erro final:", round(squaredError, 6), "\n")
  return(list(model = model, epochs = epochs, errorVec = errorVec))
}

# -----------------------------------------------------------------
# Predição para UM exemplo
# -----------------------------------------------------------------
mlp.test <- function(model, example) {
  res <- mlp.forward(model = model, example = example)
  return(res$fnet.output)
}

# -----------------------------------------------------------------
# Predição para MÚLTIPLOS exemplos
# -----------------------------------------------------------------
mlp.predict <- function(model, test.set) {
  predictions <- matrix(0, nrow = nrow(test.set), ncol = model$output.length)
  
  for(i in 1:nrow(test.set)) {
    example <- as.numeric(test.set[i, 1:model$input.length])
    pred <- mlp.test(model, example)
    predictions[i, ] <- pred
  }
  
  return(predictions)
}

# -----------------------------------------------------------------
# Função para calcular acurácia
# -----------------------------------------------------------------
mlp.accuracy <- function(model, test.set, true.classes) {
  predictions <- mlp.predict(model, test.set)
  
  if(model$output.length == 1) {
    # Caso binário
    predicted_classes <- round(predictions)
  } else {
    # Caso multiclasse
    predicted_classes <- apply(predictions, 1, which.max) - 1
  }
  
  accuracy <- sum(predicted_classes == true.classes) / length(true.classes)
  return(accuracy)
}
