//#########################################
//#########################################
// Path is relative to packages after nuget

open System
open System.IO

// 
let rootDir =  __SOURCE_DIRECTORY__ |> Directory.GetParent |> string |> Directory.GetParent |> string |> Directory.GetParent |> string 

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + rootDir )
  
let dependencies = 
    [
    "./Newtonsoft.Json/lib/netstandard2.0/"
    "./System.Memory/lib/netstandard2.0/"
    "./System.Collections.Immutable/lib/netstandard2.0/"
    "./System.Numerics.Vectors/lib/netstandard2.0/"
    "./Microsoft.Data.DataView/lib/netstandard2.0/"
    "./Microsoft.ML/lib/netstandard2.0/"
    "./Microsoft.ML.CpuMath/lib/netstandard2.0/"
    "./system.runtime.compilerservices.unsafe/lib/netstandard2.0/"
    "./Microsoft.ML.CpuMath/runtimes/win-x64/native"
    "./Microsoft.ML/runtimes/win-x64/native"
    "./System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/"
    ]

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(rootDir,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    

// Reference directories to automatically resolve dependencies (e.g. on native .dlls)
#I "../../../Microsoft.ML/runtimes/win-x64/native"
#I "../../../Microsoft.ML.CpuMath/runtimes/win-x64/native"
#I "../../../System.Memory/lib/netstandard2.0/"
#I "../../../System.Collections.Immutable/lib/netstandard2.0"
#I "../../../System.Numerics.Vectors/lib/netstandard2.0/"
#I "../../../Microsoft.Data.DataView/lib/netstandard2.0/"
#I "../../../Microsoft.ML/lib/netstandard2.0/"
#I "../../../system.runtime.compilerservices.unsafe/lib/netstandard2.0/"
#I "../../../Microsoft.ML.CpuMath/lib/netstandard2.0/"

// Reference .dlls
// Note: referencing by path is needed also when referencing the directory.
#I "../../.."
#I "../../../Newtonsoft.Json/lib/netstandard2.0"
#r "../../../Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"
#r "../../../System.Threading.Tasks.Dataflow/lib/netstandard2.0/System.Threading.Tasks.Dataflow.dll"
#r "../../../Microsoft.ML.DataView/lib/netstandard2.0/Microsoft.ML.DataView.dll"
#r "../../../System.Collections.Immutable/lib/netstandard2.0/System.Collections.Immutable.dll"
#r "../../../System.Numerics.Vectors/lib/netstandard2.0/System.Numerics.Vectors.dll"
#r "../../../System.Memory/lib/netstandard2.0/System.Memory.dll"

#r "../../../Microsoft.ML.CpuMath/lib/netstandard2.0/Microsoft.ML.CpuMath.dll"

#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Core.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Data.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Ensemble.dll"
#r "../../../Microsoft.ML.FastTree/lib/netstandard2.0/Microsoft.ML.FastTree.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.KMeansClustering.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Maml.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.PCA.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.ResultProcessor.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.SamplesUtils.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.StandardLearners.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Sweeper.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Transforms.dll"
#r "../../../System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"

#r "FSharpML.dll"

//#I @"Microsoft.ML/runtimes/win-x64/native/"



