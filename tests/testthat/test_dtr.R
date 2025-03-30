test_that("The dtr classifier works properly on the iris data set", {
  require('MASS')
  data(iris)
  set.seed(42)
  n <- nrow(iris)
  X <- as.matrix(iris[, -5])
  Y <- iris$Species
  
  
  dtr_out <- dtr(X, Y)
  predicted <- predict(dlda_out, iris[-train, -5])
  
  dlda_out2 <- dlda(x = iris[train, -5], y = iris[train, 5])
  predicted2 <- predict(dlda_out2, iris[-train, -5])
  
  # Tests that the same labels result from the matrix and formula versions of
  # the DLDA classifier
  expect_equal(predicted$class, predicted2$class)
  
  expect_is(predicted$posterior, "matrix")
})