library(tictoc)
library(functional)
library(cluster)

load("/data/mdp14vu/RF/big_data_traintest.Rdata")

all_f <- ncol(big_data_training)

big_data <- rbind(big_data_training, big_data_testing)
class_vec <- c(class_vec_training, class_vec_testing)

a_run <- function(feature_number){
  print(paste("Trying for", feature_number))
  the_data <- big_data[,1:feature_number]

  tic()
  
  tryCatch({a_model <- pam(the_data, 2)},
           error=function(e) {print(paste("Hey, this happened:",e))}
  )
  
  timed <- toc()
  if(!exists("a_model")){
    a_model <- NA
    a_score <- NA
  }else{
    a_table <- table(class_vec, a_model$clustering)
    f1_1 <- (2*a_table["0",1])/(2*a_table["0",1] + a_table["0",2] + a_table["1",1])
    f1_2 <- (2*a_table["0",2])/(a_table["0",1] + 2*a_table["0",2] + a_table["1",2])
    a_score <- max(f1_1, f1_2)
  }
  
  return(list(timed$toc-timed$tic, feature_number, a_score, a_model))
  
}

funs <- sapply(c(0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1), function(divisor) Curry(`*`,divisor) )

res_run_list <- sapply(sapply(funs, function(f) as.integer(f(all_f))), a_run)

save("res_run_list", file="/data/mdp14vu/kmeans/res_run_list_kmeans.Rdata")
