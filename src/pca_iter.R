library(tictoc)
library(functional)
library(cluster)

load("/data/mdp14vu/RF/big_data_traintest.Rdata")

all_f <- ncol(big_data_training)

big_data <- rbind(big_data_training, big_data_testing)
class_vec <- c(class_vec_training, class_vec_testing)

a_run <- function(feature_number){
  print(paste("Trying for", feature_number))
  the_data <- big_data[,1:feature_number] #change to vars?
  
  tic()
  
  tryCatch({a_model <- prcomp(the_data, center=TRUE, scale. = TRUE)},
           error=function(e) {print(paste("Hey, this happened:",e))}
  )
  
  timed <- toc()
  if(!exists("a_model")){
    a_model <- NA
    a_score <- NA
  }else{
    a_score <- (-1)
  }
  
  return(list(timed$toc-timed$tic, feature_number, a_score, a_model))
  
}

funs <- sapply(c(0.2,0.25,0.3,0.4,0.5,0.6,0.7,0.75,0.8,0.9,1), function(divisor) Curry(`*`,divisor) )

res_run_list <- sapply(sapply(funs, function(f) as.integer(f(all_f))), a_run)

save("res_run_list", file="/data/mdp14vu/pca/res_run_list_pca.Rdata")
