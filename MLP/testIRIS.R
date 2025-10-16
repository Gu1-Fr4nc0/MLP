# ==================================================================
# ARQUIVO: testIRIS.R  
# DESCRIÇÃO: Testa o MLP com Iris completo [4, H, 3]
# ==================================================================

install.packages("ggplot2")
library(ggplot2)
source("mlp.R")

# Configuração
seeds <- c(42, 123, 456)
results <- list()

cat("=== TESTE MLP COM IRIS (3 CLASSES) - VERSÃO CORRIGIDA ===\n")

# Preparar dados Iris
data(iris)
iris_data <- iris
iris_data$class <- as.numeric(factor(iris$Species)) - 1  # 0,1,2

# Normalizar features
normalize <- function(x) { (x - min(x)) / (max(x) - min(x)) }
iris_data[, 1:4] <- as.data.frame(lapply(iris_data[, 1:4], normalize))

# Codificar one-hot manualmente
iris_encoded <- data.frame(
  Sepal.Length = iris_data$Sepal.Length,
  Sepal.Width = iris_data$Sepal.Width,
  Petal.Length = iris_data$Petal.Length,
  Petal.Width = iris_data$Petal.Width,
  class_0 = as.numeric(iris_data$class == 0),
  class_1 = as.numeric(iris_data$class == 1),
  class_2 = as.numeric(iris_data$class == 2)
)

cat("Dimensões do dataset:", dim(iris_encoded), "\n")
cat("Primeiras linhas:\n")
print(head(iris_encoded))

pdf("mlp_iris_comparison.pdf", width = 12, height = 10)

for(i in 1:length(seeds)) {
  cat("\n--- Execução", i, "com seed", seeds[i], "---\n")
  
  tryCatch({
    # MLP [4, 5, 3]
    model <- mlp.create(input.length = 4, hidden.length = 5, output.length = 3)
    trained <- mlp.train(model, iris_encoded, lrn.rate = 0.1, threshold = 0.05, n.iter = 1000)
    
    # Avaliar usando a nova função
    accuracy <- mlp.accuracy(trained$model, iris_encoded[, 1:4], iris_data$class)
    
    results[[i]] <- list(
      seed = seeds[i],
      accuracy = accuracy,
      trained = trained
    )
    
    cat("✅ Acurácia:", round(accuracy, 4), "\n")
    
    # Gráfico (apenas se treinamento foi bem sucedido)
    if(length(trained$errorVec) > 0) {
      performance_data <- data.frame(
        Epoca = 1:trained$epochs,
        Erro = trained$errorVec,
        Execucao = paste("Seed", seeds[i])
      )
      
      p <- ggplot(performance_data, aes(x = Epoca, y = Erro, color = Execucao)) +
        geom_line() +
        labs(title = paste("MLP Iris - Seed", seeds[i]),
             subtitle = paste("Acurácia:", round(accuracy, 4)),
             x = "Época", y = "Erro Quadrático Médio") +
        theme_minimal()
      
      print(p)
    }
    
  }, error = function(e) {
    cat("❌ Erro na execução", i, ":", e$message, "\n")
  })
}

# Gráfico comparativo (apenas se temos resultados)
if(length(results) > 0) {
  comparison_data <- data.frame()
  
  for(i in 1:length(results)) {
    if(!is.null(results[[i]])) {
      temp_data <- data.frame(
        Epoca = 1:results[[i]]$trained$epochs,
        Erro = results[[i]]$trained$errorVec,
        Execucao = paste("Seed", seeds[i])
      )
      comparison_data <- rbind(comparison_data, temp_data)
    }
  }
  
  if(nrow(comparison_data) > 0) {
    p_final <- ggplot(comparison_data, aes(x = Epoca, y = Erro, color = Execucao)) +
      geom_line(alpha = 0.7) +
      labs(title = "Comparação MLP Iris - Múltiplas Seeds",
           x = "Época", y = "Erro Quadrático Médio") +
      theme_minimal()
    
    print(p_final)
  }
}

dev.off()

# Resultados finais
cat("\n=== RESULTADOS FINAIS IRIS ===\n")
if(length(results) > 0) {
  for(i in 1:length(results)) {
    if(!is.null(results[[i]])) {
      cat("Seed", seeds[i], "- Acurácia:", round(results[[i]]$accuracy, 4), 
          "- Épocas:", results[[i]]$trained$epochs, "\n")
    }
  }
} else {
  cat("Nenhum resultado foi gerado devido a erros.\n")
}
