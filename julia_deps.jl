using Pkg

for (pkg, path) in [
    ("igraph_jll", "/nix/store/p89x11x3nb62b9qvd57rgyghpkvggnvl-source"),
    ("leiden_jll", "/nix/store/5hyzr4d5nj51mvii224n0z1dw14ywnir-source"),
    ("Leiden", "/nix/store/611wgj6ynw0knrk6jrnjdc4q935pl83j-source"),
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

