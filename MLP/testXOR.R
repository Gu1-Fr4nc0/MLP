# ==================================================================
# ARQUIVO: testXOR.R
# DESCRIÇÃO: Testa o MLP com dataset XOR [2, H, 1]
# ==================================================================

install.packages("ggplot2")
library(ggplot2)
source("mlp.R")

# Configuração de seeds
seeds <- c(42, 123, 456, 789, 999)
results <- list()

cat("=== TESTE MLP COM XOR ===\n")

# Dataset XOR
xor_data <- data.frame(
  x1 = c(0, 0, 1, 1),
  x2 = c(0, 1, 0, 1),
  class = c(0, 1, 1, 0)
)

pdf("mlp_xor_comparison.pdf", width = 12, height = 10)

for(i in 1:length(seeds)) {
  cat("\n--- Execução", i, "com seed", seeds[i], "---\n")
  
  # Criar e treinar MLP [2, 4, 1]
  model <- mlp.create(input.length = 2, hidden.length = 4, output.length = 1)
  trained <- mlp.train(model, xor_data, lrn.rate = 0.3, threshold = 0.01, n.iter = 5000)
  
  # Fazer predições
  predictions <- mlp.predict(trained$model, xor_data)
  predicted_classes <- round(predictions)
  accuracy <- sum(predicted_classes == xor_data$class) / nrow(xor_data)
  
  results[[i]] <- list(
    seed = seeds[i],
    model = trained$model,
    trained = trained,
    accuracy = accuracy,
    predictions = predicted_classes
  )
  
  # Gráfico de performance
  performance_data <- data.frame(
    Epoca = 1:trained$epochs,
    Erro = trained$errorVec,
    Execucao = paste("Seed", seeds[i])
  )
  
  p <- ggplot(performance_data, aes(x = Epoca, y = Erro, color = Execucao)) +
    geom_line() +
    labs(title = paste("MLP XOR - Seed", seeds[i]),
         subtitle = paste("Acurácia:", round(accuracy, 4)),
         x = "Época", y = "Erro Quadrático Médio") +
    theme_minimal()
  
  print(p)
}

# Gráfico comparativo final
comparison_data <- data.frame()
for(i in 1:length(results)) {
  temp_data <- data.frame(
    Epoca = 1:results[[i]]$trained$epochs,
    Erro = results[[i]]$trained$errorVec,
    Execucao = paste("Seed", seeds[i])
  )
  comparison_data <- rbind(comparison_data, temp_data)
}

p_final <- ggplot(comparison_data, aes(x = Epoca, y = Erro, color = Execucao)) +
  geom_line(alpha = 0.7) +
  labs(title = "Comparação MLP XOR - Múltiplas Seeds",
       x = "Época", y = "Erro Quadrático Médio") +
  theme_minimal()

print(p_final)

# Resultados finais
cat("\n=== RESULTADOS FINAIS XOR ===\n")
for(i in 1:length(results)) {
  cat("Seed", seeds[i], "- Acurácia:", round(results[[i]]$accuracy, 4), 
      "- Épocas:", results[[i]]$trained$epochs, "\n")
}

dev.off()
