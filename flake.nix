{
  description = "Flake for DeePWAK.jl";
  nixConfig = { bash-prompt = "\[DeePWAK$(__git_ps1 \" (%s)\")\]$ "; };

  inputs = {
    utils.url = "github:numtide/flake-utils";
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable"; # Or pin to match CrobustaScreen
	#nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11"; # Or other stable branch

    # --- DeePWAK's Julia Dependencies ---
    # Non-flake source inputs (renamed with Src suffix)
	/*
    igraph_jllSrc = { url = "github:fcdimitr/igraph_jll.jl"; flake = false; };
    leiden_jllSrc = { url = "github:fcdimitr/leiden_jll.jl"; flake = false; };
    LeidenSrc = { url = "github:pitsianis/Leiden.jl"; flake = false; };
	*/
    # Flake inputs providing Nix packages
    Autoencoders = { url = "github:kewiechecki/Autoencoders.jl"; flake = true; inputs.nixpkgs.follows = "nixpkgs"; };
    DictMap = { url = "github:kewiechecki/DictMap.jl"; flake = true; inputs.nixpkgs.follows = "nixpkgs"; };
    TrainingIO = { url = "github:kewiechecki/TrainingIO.jl"; flake = true; inputs.nixpkgs.follows = "nixpkgs"; };
  };

  outputs = { 
    self, nixpkgs, utils,
	#igraph_jllSrc, leiden_jllSrc, LeidenSrc,
	Autoencoders, DictMap, TrainingIO
  }@inputs: # Use @inputs syntax
    utils.lib.eachDefaultSystem (system:
      let
        # --- Julia Package Overlay ---
        # Defines how Nix finds/builds all custom Julia deps needed by DeePWAK
		/*
        juliaOverlay = final: prev: {
          juliaPackages = prev.juliaPackages // { # <<< Standard // merge
            # Build packages from source inputs
            igraph_jll = final.juliaPackages.buildJuliaPackage {
               pname = "igraph_jll"; version = "git"; src = inputs.igraph_jllSrc;
            };
            leiden_jll = final.juliaPackages.buildJuliaPackage {
              pname = "leiden_jll"; version = "git"; src = inputs.leiden_jllSrc;
            };
            Leiden = final.juliaPackages.buildJuliaPackage {
              pname = "Leiden"; version = "git"; src = inputs.LeidenSrc;
            };
            # Import packages directly from flake input Nix outputs
            Autoencoders = inputs.Autoencoders.packages.${system}.default;
            DictMap = inputs.DictMap.packages.${system}.default;
            TrainingIO = inputs.TrainingIO.packages.${system}.default;
          };
        };
		*/

        # --- Base Nixpkgs with Overlay ---
        pkgs = import nixpkgs {
          inherit system;
          #overlays = [ juliaOverlay ]; # <<< Apply the overlay
          config = { allowUnfree = true; cudaSupport = system == "x86_64-linux"; };
        };
 
        # --- Access Overlay Packages by Name ---
        # Get handles to the Nix derivations defined/imported via the overlay.
        # We can now use these variables below.
		/*
        igraph_jllPkg = pkgs.juliaPackages.igraph_jll;
        leiden_jllPkg = pkgs.juliaPackages.leiden_jll;
        leidenPkg = pkgs.juliaPackages.Leiden;
		*/
        autoencodersPkg = pkgs.juliaPackages.Autoencoders;
        dictMapPkg = pkgs.juliaPackages.DictMap;
        trainingIOPkg = pkgs.juliaPackages.TrainingIO;

        # Extract juliaPackages after overlay for convenience
        juliaPkgs = pkgs.juliaPackages;

        # --- Build DeePWAK.jl ---
        # Assumes DeePWAK/Project.toml lists dependencies like "Leiden", "Autoencoders", etc.
        # buildJuliaPackage uses the overlay-enhanced pkgs to find them.
		/*
        deePWAKbuilt = juliaPkgs.buildJuliaPackage {
          pname = "DeePWAK";
          version = "0.1.2";
          src = ./.; # DeePWAK source code

          # Propagate only necessary SYSTEM runtime libs
          propagatedBuildInputs = [
            pkgs.gfortran         # For libquadmath etc. if needed runtime
            pkgs.stdenv.cc.cc.lib # For libstdc++, libgcc_s etc. runtime
            # Add CUDA libs if needed by DeePWAK itself at runtime
            pkgs.cudaPackages.cudnn
            igraph_jllPkg
            leiden_jllPkg
            leidenPkg
			autoencodersPkg
            dictMapPkg
            trainingIOPkg
          ];
        };
		*/

        # --- Shell Environment Packages ---
        shellPkgsNested = with pkgs; [
          julia git stdenv.cc gfortran stdenv.cc.cc.lib
          (lib.optional stdenv.isLinux cudaPackages.cudatoolkit)
          (lib.optional stdenv.isLinux cudaPackages.cudnn)

          # Add deePWAKbuilt here if you want the nix-built version easily importable in shell
          # deePWAKbuilt
        ];
        shellPkgs = pkgs.lib.flatten shellPkgsNested;

      in {
        # Export the built DeePWAK package
        #packages.deePWAK = deePWAKbuilt;
        #packages.default = self.packages.${system}.deePWAK;

        # Development Shell
        devShell = pkgs.mkShell { # Use pkgs. explicitly
          name = "deepwak-dev-shell";
          buildInputs = shellPkgs;

          # --- Cleaned shellHook ---
          shellHook = ''
            source ${pkgs.git}/share/bash-completion/completions/git-prompt.sh
            # Point Julia to local source; dependencies come from Nix build via overlay
            export JULIA_PROJECT="@."

            # Set LD_LIBRARY_PATH for system libs
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath shellPkgs}";

            echo "Nix dev shell for DeePWAK.jl activated."
            echo "Julia environment uses Project.toml (JULIA_PROJECT=@.). Deps via Nix overlay."
            # --- NO MORE julia_deps.jl ---
          '';
        };
      }
    );
}
