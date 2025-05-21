{
  description = "Flake for DeePWAK.jl";
  nixConfig = {
    bash-prompt = "\[DeePWAK$(__git_ps1 \" (%s)\")\]$ ";
  };

  inputs = {
    flake-utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

	/*
    igraph_jll = {
      url = "github:fcdimitr/igraph_jll.jl";
      flake = false;
    };
    leiden_jll = {
      url = "github:fcdimitr/leiden_jll.jl";
      flake = false;
    };
    Leiden = {
      url = "github:pitsianis/Leiden.jl";
      flake = false;
    };
	*/

    Autoencoders = {
      url = "github:kewiechecki/Autoencoders.jl";
      flake = true;
      inputs.nixpkgs.follows = "nixpkgs";
    };
    DictMap = {
      url = "github:kewiechecki/DictMap.jl";
      flake = true;
      inputs.nixpkgs.follows = "nixpkgs";
    };
    TrainingIO = {
      url = "github:kewiechecki/TrainingIO.jl";
      flake = true;
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = { self, nixpkgs,flake-utils, #igraph_jll, leiden_jll, Leiden, 
  			  Autoencoders, DictMap, TrainingIO }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system;
          config.allowUnfree = true;
          config.cudaSupport = system == "x86_64-linux";
		};

        juliaPkgs = pkgs.juliaPackages;
        shellPkgsNested = with pkgs; [ # Keep shell packages definition separate
          julia git stdenv.cc gfortran stdenv.cc.cc.lib
          (lib.optional stdenv.isLinux cudaPackages.cudatoolkit)
          (lib.optional stdenv.isLinux cudaPackages.cudnn)
        ];
        shellPkgs = pkgs.lib.flatten shellPkgsNested;

        # --- Get the actual DictMap Nix package derivation ---
        # Assumes the DictMap flake exports packages.default correctly
        autoencodersPkg = Autoencoders.packages.${system}.default;
        dictMapPkg = DictMap.packages.${system}.default;
        trainingIOPkg = TrainingIO.packages.${system}.default;

        # Build DeePWAK.jl
        deePWAKbuilt = juliaPkgs.buildJuliaPackage {
          pname = "DeePWAK";
          version = "0.1.1"; # TODO: FIX THIS
          src = ./.;

          # Propagate runtime libs (gfortan) AND the dependency package (dictMapPkg)
          # This signals that users/builders of TrainingIO need these.
          # The Nix Julia hooks MIGHT use this to make dictMapPkg available
          # to the Julia build environment for TrainingIO.
          propagatedBuildInputs = [
            pkgs.gfortran
			autoencodersPkg
            dictMapPkg
			trainingIOPkg
          ];
        };

        # Get library paths from the stdenv compiler and from gfortran.
        gccPath = toString pkgs.stdenv.cc.cc.lib;
        gfortranPath = toString pkgs.gfortran;

        # Define the multi-line Julia script.
        # NOTE: The closing delimiter (two single quotes) MUST be flush with the left margin.
		/*
        juliaScript = ''
using Pkg

for (pkg, path) in [
    ("igraph_jll", "__IGRAPH_JLL__"),
    ("leiden_jll", "__LEIDEN_JLL__"),
    ("Leiden", "__LEIDEN__"),
]
    try
        @eval import __DOLLAR_PLACEHOLDER__(Symbol(pkg))
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
'';
*/
		
      in {
        # A derivation for your package.
        packages.deePWAK = deePWAKbuilt;
        packages.default = self.packages.${system}.deePWAK;

        # A development shell that provides Julia with your package instantiated.
        devShell = with pkgs; mkShell {
          name = "deepwak-dev-shell";
          buildInputs = shellPkgs;

          shellHook = ''
source ${git}/share/bash-completion/completions/git-prompt.sh
export JULIA_PROJECT="@."
export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath shellPkgs}";

echo "Nix dev shell for TrainingIO.jl activated." # Corrected package name
echo "Julia environment uses Project.toml (JULIA_PROJECT=@.)."
# Debug check
echo "--- Checking LD_LIBRARY_PATH ($LD_LIBRARY_PATH) for libquadmath.so.0 ---"
( IFS=: ; for p in $LD_LIBRARY_PATH; do if [ -f "$p/libquadmath.so.0" ]; then echo "  FOUND in $p"; fi; done )
echo "----------------------------------------------------"
'';
/*

cat > julia_deps.jl <<'EOF'
${juliaScript}
EOF

# Replace placeholders with actual paths.
sed -i 's|__IGRAPH_JLL__|${toString igraph_jll}|g' julia_deps.jl
sed -i 's|__LEIDEN_JLL__|${toString leiden_jll}|g' julia_deps.jl
sed -i 's|__LEIDEN__|${toString Leiden}|g' julia_deps.jl

# Replace the dollar placeholder with a literal dollar sign.
sed -i 's|__DOLLAR_PLACEHOLDER__|\\$|g' julia_deps.jl

# Activate the project and instantiate dependencies.
julia --project=. julia_deps.jl
'';
*/
        };
      }
    );
}

