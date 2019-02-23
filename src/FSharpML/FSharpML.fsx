#r @"netstandard.dll"

open System
open System.IO

// Path is relative to bin/FSharpML/netstandard2.0 (main bin folder)
let rootDir =  __SOURCE_DIRECTORY__ |> Directory.GetParent |> string |> Directory.GetParent |> string |> Directory.GetParent |> string 

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + rootDir )
  
let dependencies = 
    [
    "./packages/Newtonsoft.Json/lib/netstandard2.0/"
    "./packages/System.Memory/lib/netstandard2.0/"
    "./packages/System.Collections.Immutable/lib/netstandard2.0/"
    "./packages/System.Numerics.Vectors/lib/netstandard2.0/"
    "./packages/Microsoft.Data.DataView/lib/netstandard2.0/"
    "./packages/Microsoft.ML/lib/netstandard2.0/"
    "./packages/Microsoft.ML.CpuMath/lib/netstandard2.0/"
    "./packages/system.runtime.compilerservices.unsafe/lib/netstandard2.0/"
    "./packages/Microsoft.ML.CpuMath/runtimes/win-x64/native"
    "./packages/Microsoft.ML/runtimes/win-x64/native"
    "./packages/System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/"
    ]

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(rootDir,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    

// Reference directories to automatically resolve dependencies (e.g. on native .dlls)
#I "../../../packages/Microsoft.ML/runtimes/win-x64/native"
#I "../../../packages/Microsoft.ML.CpuMath/runtimes/win-x64/native"
#I "../../../packages/System.Memory/lib/netstandard2.0/"
#I "../../../packages/System.Collections.Immutable/lib/netstandard2.0"
#I "../../../packages/System.Numerics.Vectors/lib/netstandard2.0/"
#I "../../../packages/Microsoft.Data.DataView/lib/netstandard2.0/"
#I "../../../packages/Microsoft.ML/lib/netstandard2.0/"
#I "../../../packages/system.runtime.compilerservices.unsafe/lib/netstandard2.0/"
#I "../../../packages/Microsoft.ML.CpuMath/lib/netstandard2.0/"

// Reference .dlls
// Note: referencing by path is needed also when referencing the directory.
#I "../../../packages"
#I "../../../packages/Newtonsoft.Json/lib/netstandard2.0"
#r "../../../packages/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"
#r "../../../packages/System.Threading.Tasks.Dataflow/lib/netstandard2.0/System.Threading.Tasks.Dataflow.dll"
#r "../../../packages/Microsoft.Data.DataView/lib/netstandard2.0/Microsoft.Data.DataView.dll"
#r "../../../packages/System.Collections.Immutable/lib/netstandard2.0/System.Collections.Immutable.dll"
#r "../../../packages/System.Numerics.Vectors/lib/netstandard2.0/System.Numerics.Vectors.dll"
#r "../../../packages/System.Memory/lib/netstandard2.0/System.Memory.dll"

#r "../../../packages/Microsoft.ML.CpuMath/lib/netstandard2.0/Microsoft.ML.CpuMath.dll"

#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Core.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Data.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Ensemble.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.FastTree.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.KMeansClustering.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Maml.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.PCA.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.ResultProcessor.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.SamplesUtils.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.StandardLearners.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Sweeper.dll"
#r "../../../packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Transforms.dll"
#r "../../../packages/System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"

#r "../../../bin/FSharpML/netstandard2.0/FSharpML.dll"

//#I @"Microsoft.ML/runtimes/win-x64/native/"



