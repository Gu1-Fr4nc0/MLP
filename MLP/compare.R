# ==================================================================
# ARQUIVO: compare.R
# DESCRIÇÃO: Compara a implementação com pacotes do R
# ==================================================================

install.packages("neuralnet")
install.packages("RSNNS")
install.packages("ggplot2")
library(neuralnet)
library(RSNNS)
library(ggplot2)
source("mlp.R")

cat("=== COMPARAÇÃO ENTRE IMPLEMENTAÇÕES ===\n")

# Dataset XOR
xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  class = c(0, 1, 1, 0)
)

results_comparison <- list()

# 1. IMPLEMENTAÇÃO
cat("\n1. IMPLEMENTAÇÃO MLP:\n")
our_model <- mlp.create(input.length = 2, hidden.length = 4, output.length = 1)
our_trained <- mlp.train(our_model, xor_data, lrn.rate = 0.3, threshold = 0.01, n.iter = 5000)
our_predictions <- round(mlp.predict(our_trained$model, xor_data))
our_accuracy <- sum(our_predictions == xor_data$class) / nrow(xor_data)

results_comparison[["MLP"]] <- list(
  accuracy = our_accuracy,
  epochs = our_trained$epochs,
  final_error = tail(our_trained$errorVec, 1)
)

cat("✅MLP - Acurácia:", our_accuracy, "- Épocas:", our_trained$epochs, "\n")

# 2. PACOTE NEURALNET (CORRIGIDO)
cat("\n2. PACOTE NEURALNET:\n")
tryCatch({
  # Neuralnet precisa de dados ligeiramente diferentes
  nn_data <- data.frame(
    x1 = c(0, 0, 1, 1),
    x2 = c(0, 1, 0, 1),
    class = c(0, 1, 1, 0)
  )
  
  # Para neuralnet, é melhor usar mais épocas e taxa de aprendizado menor
  nn_model <- neuralnet(class ~ x1 + x2, 
                        data = nn_data,
                        hidden = 4, 
                        linear.output = FALSE,  # IMPORTANTE para classificação
                        learningrate = 0.1,     # Taxa menor
                        threshold = 0.01,
                        stepmax = 10000,        # Mais épocas
                        lifesign = "full")
  
  nn_predictions <- round(predict(nn_model, nn_data[, 1:2]))
  nn_accuracy <- sum(nn_predictions == nn_data$class) / nrow(nn_data)
  
  results_comparison[["neuralnet"]] <- list(
    accuracy = nn_accuracy,
    epochs = nn_model$result.matrix[["steps"]],
    final_error = nn_model$result.matrix[["error"]]
  )
  
  cat("✅ neuralnet - Acurácia:", nn_accuracy, "- Épocas:", nn_model$result.matrix[["steps"]], "\n")
  
}, error = function(e) {
  cat("❌ neuralnet falhou:", e$message, "\n")
  # Tentar abordagem alternativa
  tryCatch({
    cat("Tentando abordagem alternativa para neuralnet...\n")
    nn_data <- data.frame(
      x1 = c(0, 0, 1, 1),
      x2 = c(0, 1, 0, 1),
      class = c(0, 1, 1, 0)
    )
    
    nn_model <- neuralnet(class ~ x1 + x2, 
                          data = nn_data,
                          hidden = 2,  # Menos neurônios
                          linear.output = FALSE,
                          act.fct = "logistic",
                          stepmax = 2000)
    
    nn_predictions <- round(predict(nn_model, nn_data[, 1:2]))
    nn_accuracy <- sum(nn_predictions == nn_data$class) / nrow(nn_data)
    
    results_comparison[["neuralnet"]] <- list(
      accuracy = nn_accuracy,
      epochs = nn_model$result.matrix[["steps"]],
      final_error = nn_model$result.matrix[["error"]]
    )
    
    cat("✅ neuralnet (alternativo) - Acurácia:", nn_accuracy, "\n")
  }, error = function(e2) {
    cat("❌ neuralnet falhou completamente:", e2$message, "\n")
  })
})

# 3. PACOTE RSNNS (CORRIGIDO)
cat("\n3. PACOTE RSNNS:\n")
tryCatch({
  inputs <- as.matrix(xor_data[, 1:2])
  targets <- decodeClassLabels(xor_data$class)
  
  # RSNNS funciona melhor com normalização
  inputs <- normalizeData(inputs, type = "0_1")
  
  rsnns_model <- mlp(inputs, targets, 
                     size = 4,
                     learnFunc = "BackpropBatch",
                     learnFuncParams = c(0.1, 0.1),  # learning rate e momentum
                     maxit = 5000,
                     linOut = FALSE)  # Para classificação
  
  rsnns_predictions <- predict(rsnns_model, inputs)
  rsnns_class_predictions <- encodeClassLabels(rsnns_predictions)
  rsnns_accuracy <- sum(rsnns_class_predictions - 1 == xor_data$class) / nrow(xor_data)
  
  results_comparison[["RSNNS"]] <- list(
    accuracy = rsnns_accuracy,
    epochs = rsnns_model$fittedIterations,
    final_error = rsnns_model$fittedError
  )
  
  cat("✅ RSNNS - Acurácia:", rsnns_accuracy, "- Épocas:", rsnns_model$fittedIterations, "\n")
  
}, error = function(e) {
  cat("❌ RSNNS falhou:", e$message, "\n")
})

# 4. RESULTADOS DA COMPARAÇÃO
cat("\n")
cat("==================================================\n")
cat("RESULTADOS DA COMPARAÇÃO - DATASET XOR\n")
cat("==================================================\n")

if(length(results_comparison) > 0) {
  comparison_df <- data.frame(
    Implementacao = names(results_comparison),
    Acuracia = sapply(results_comparison, function(x) round(x$accuracy, 4)),
    Epocas = sapply(results_comparison, function(x) ifelse(is.null(x$epochs), NA, x$epochs)),
    Erro_Final = sapply(results_comparison, function(x) round(x$final_error, 4))
  )
  
  print(comparison_df)
} else {
  cat("Nenhuma implementação funcionou para comparação.\n")
}

# 5. GRÁFICOS COMPARATIVOS
pdf("comparacao_implementacoes_fixed.pdf", width = 12, height = 10)

# Gráfico de barras - Acurácia
if(length(results_comparison) > 0) {
  accuracy_data <- data.frame(
    Implementacao = names(results_comparison),
    Acuracia = sapply(results_comparison, function(x) x$accuracy)
  )
  
  p1 <- ggplot(accuracy_data, aes(x = reorder(Implementacao, -Acuracia), y = Acuracia, fill = Implementacao)) +
    geom_bar(stat = "identity", alpha = 0.8) +
    geom_text(aes(label = paste0(round(Acuracia * 100, 1), "%")), vjust = -0.5, size = 5) +
    labs(title = "Comparação de Acurácia entre Implementações MLP",
         subtitle = "Problema XOR (4 padrões)",
         x = "Implementação", y = "Acurácia") +
    ylim(0, 1.1) +
    theme_minimal() +
    theme(legend.position = "none",
          axis.text = element_text(size = 10),
          plot.title = element_text(size = 16, face = "bold"))
  
  print(p1)
  
  # Curva de aprendizado do nosso MLP
  error_data <- data.frame(
    Epoca = 1:our_trained$epochs,
    Erro = our_trained$errorVec
  )
  
  p2 <- ggplot(error_data, aes(x = Epoca, y = Erro)) +
    geom_line(color = "blue", linewidth = 1) +
    geom_hline(yintercept = 0.01, linetype = "dashed", color = "red", alpha = 0.7) +
    annotate("text", x = max(error_data$Epoca)/2, y = 0.015, 
             label = "Threshold = 0.01", color = "red") +
    labs(title = "Curva de Aprendizado - Nossa Implementação MLP",
         subtitle = paste("Épocas:", our_trained$epochs, "- Acurácia Final:", round(our_accuracy * 100, 1), "%"),
         x = "Época", y = "Erro Quadrático Médio") +
    theme_minimal()
  
  print(p2)
  
  # Comparação de épocas (se disponível)
  epochs_data <- data.frame(
    Implementacao = names(results_comparison),
    Epocas = sapply(results_comparison, function(x) ifelse(is.null(x$epochs), NA, x$epochs))
  )
  epochs_data <- epochs_data[!is.na(epochs_data$Epocas), ]
  
  if(nrow(epochs_data) > 0) {
    p3 <- ggplot(epochs_data, aes(x = reorder(Implementacao, -Epocas), y = Epocas, fill = Implementacao)) +
      geom_bar(stat = "identity", alpha = 0.8) +
      geom_text(aes(label = Epocas), vjust = -0.5, size = 5) +
      labs(title = "Comparação de Épocas para Convergência",
           x = "Implementação", y = "Número de Épocas") +
      theme_minimal() +
      theme(legend.position = "none")
    
    print(p3)
  }
  
} else {
  # Gráfico vazio se não há resultados
  p_empty <- ggplot() + 
    annotate("text", x = 1, y = 1, label = "Nenhuma implementação funcionou", size = 8) +
    theme_void()
  print(p_empty)
}

dev.off()
