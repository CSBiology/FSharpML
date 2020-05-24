//#########################################
//#########################################
// Path is relative to packages after nuget

open System
open System.IO
open System.Runtime.InteropServices

// 
let rootDir =  __SOURCE_DIRECTORY__ |> Directory.GetParent |> string |> Directory.GetParent |> string |> Directory.GetParent |> string 

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + rootDir )
  
let baseDependencies = 
    [
    ""
    "./Newtonsoft.Json/lib/netstandard2.0"
    "./System.Memory/lib/netstandard2.0/"
    "./System.Collections.Immutable/lib/netstandard2.0"
    "./System.Numerics.Vectors/lib/netstandard2.0/"
    "./Microsoft.ML.DataView/lib/netstandard2.0/"
    "./Microsoft.ML/lib/netstandard2.0/"
    "./Microsoft.ML.CpuMath/lib/netstandard2.0/"
    "./system.runtime.compilerservices.unsafe/lib/netstandard2.0/"
    "./System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/"
    ]

// Reference directories to automatically resolve dependencies (e.g. on native .dlls)
let nativeWinX86 = 
    [
    "./Microsoft.ML.CpuMath/runtimes/win-x86/nativeassets/netstandard2.0/"
    "./Microsoft.ML/runtimes/win-x86/native"
    "./Microsoft.ML.FastTree/runtimes/win-x86/native"
    ]

let nativeWinX64 = 
    [
    "./Microsoft.ML.CpuMath/runtimes/win-x64/nativeassets/netstandard2.0/"
    "./Microsoft.ML/runtimes/win-x64/native"
    "./Microsoft.ML.FastTree/runtimes/win-x64/native"
    ]

let nativeLinuxX64 = 
    [
    "./Microsoft.ML.CpuMath/runtimes/linux-x64/nativeassets/netstandard2.0/"
    "./Microsoft.ML/runtimes/linux-x64/native"
    "./Microsoft.ML.FastTree/runtimes/linux-x64/native"
    ]

let nativeOSX64 = 
    [
    "./Microsoft.ML.CpuMath/runtimes/osx-x64/nativeassets/netstandard2.0/"
    "./Microsoft.ML/runtimes/osx-x64/native"
    "./Microsoft.ML.FastTree/runtimes/osx-x64/native"
    ]
    

let dependencies =
    match Environment.Is64BitProcess with
    | false -> baseDependencies @ nativeWinX86
    | true -> 
        match RuntimeInformation.IsOSPlatform(OSPlatform.Windows) with
        | true  -> baseDependencies @ nativeWinX64
        | false ->
            match RuntimeInformation.IsOSPlatform(OSPlatform.Linux) with
            | true  -> baseDependencies @ nativeLinuxX64
            | false ->
                match RuntimeInformation.IsOSPlatform(OSPlatform.OSX) with
                | true  -> baseDependencies @ nativeOSX64
                | false -> failwithf "Platform not supported."


dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(rootDir,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    

// Reference .dlls
// Note: referencing by path is needed also when referencing the directory.
#r "../../../Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"

#r "../../../Microsoft.ML.DataView/lib/netstandard2.0/Microsoft.ML.DataView.dll"
#r "../../../System.Memory/lib/netstandard2.0/System.Memory.dll"
#r "../../../System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r "../../../System.Numerics.Vectors/lib/netstandard2.0/System.Numerics.Vectors.dll"
#r "../../../Microsoft.ML.CpuMath/lib/netstandard2.0/Microsoft.ML.CpuMath.dll"
#r "../../../System.Threading.Tasks.Dataflow/lib/netstandard2.0/System.Threading.Tasks.Dataflow.dll"

#r "../../../System.Collections.Immutable/lib/netstandard2.0/System.Collections.Immutable.dll"

#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Core.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Data.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.StandardTrainers.dll"
#r "../../../Microsoft.ML.FastTree/lib/netstandard2.0/Microsoft.ML.FastTree.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.KMeansClustering.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.PCA.dll"
#r "../../../Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Transforms.dll"

#r "FSharpML.dll"


