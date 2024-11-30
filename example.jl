using DeePWAK
using TrainingIO

dat = (DataFrame ∘ CSV.File)("exampledat/z_dat.csv",normalizenames=true);

#Preprocessing

# remove row names
dat = Matrix(dat[:,2:end]);

# scale data such that all values are between -1 and 1 
X = scaledat(dat')

m,n = size(X)

# initialize model
M = DEWAKSS(X)
