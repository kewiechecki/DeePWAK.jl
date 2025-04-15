using Pkg
Pkg.instantiate()

Pkg.add("cuDNN")
Pkg.add("StructArrays")

for (pkg, path) in [
    ("igraph_jll", "/nix/store/p89x11x3nb62b9qvd57rgyghpkvggnvl-source"),
    ("leiden_jll", "/nix/store/5hyzr4d5nj51mvii224n0z1dw14ywnir-source"),
    ("Leiden", "/nix/store/611wgj6ynw0knrk6jrnjdc4q935pl83j-source"),
    ("Autoencoders", "/nix/store/vy10h53pak4ziwz0n9fl9lirhl6ydwkf-source"),
    ("DictMap", "/nix/store/bsaib7phfah203wa7h24097fn18hjfwg-source"),
    ("TrainingIO", "/nix/store/a1gsnwnlbbqy3iv9a1ywaxjbcz49qgh7-source"),
]
    try
        @eval import \$(Symbol(pkg))
        println("Package ", pkg, " is already installed.")
    catch e
        println("Developing package ", pkg, " from ", path)
        try
            Pkg.develop(path=path)
            #Pkg.precompile(only=[pkg])
        catch e
            println("Error precompiling ", pkg, ": ", e)
            #exit(1)
        end
    end
end

Pkg.update()
using DeePWAK

