#r @"netstandard.dll"

open System
open System.IO
open System.Runtime.InteropServices

Environment.SetEnvironmentVariable("Path",
    Environment.GetEnvironmentVariable("Path") + ";" + __SOURCE_DIRECTORY__ )
  
let baseDependencies = 
    [
    "./packages/"
    "./packages/Newtonsoft.Json/lib/netstandard2.0"
    "./packages/System.Memory/lib/netstandard2.0/"
    "./packages/System.Collections.Immutable/lib/netstandard2.0"
    "./packages/System.Numerics.Vectors/lib/netstandard2.0/"
    "./packages/Microsoft.ML.DataView/lib/netstandard2.0/"
    "./packages/Microsoft.ML/lib/netstandard2.0/"
    "./packages/Microsoft.ML.CpuMath/lib/netstandard2.0/"
    "./packages/system.runtime.compilerservices.unsafe/lib/netstandard2.0/"
    "./packages/System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/"
    ]

// Reference directories to automatically resolve dependencies (e.g. on native .dlls)
let nativeWinX86 = 
    [
    "./packages/Microsoft.ML.CpuMath/runtimes/win-x86/nativeassets/netstandard2.0/"
    "./packages/Microsoft.ML/runtimes/win-x86/native"
    "./packages/Microsoft.ML.FastTree/runtimes/win-x86/native"
    ]

let nativeWinX64 = 
    [
    "./packages/Microsoft.ML.CpuMath/runtimes/win-x64/nativeassets/netstandard2.0/"
    "./packages/Microsoft.ML/runtimes/win-x64/native"
    "./packages/Microsoft.ML.FastTree/runtimes/win-x64/native"
    ]

let nativeLinuxX64 = 
    [
    "./packages/Microsoft.ML.CpuMath/runtimes/linux-x64/nativeassets/netstandard2.0/"
    "./packages/Microsoft.ML/runtimes/linux-x64/native"
    "./packages/Microsoft.ML.FastTree/runtimes/linux-x64/native"
    ]

let nativeOSX64 = 
    [
    "./packages/Microsoft.ML.CpuMath/runtimes/osx-x64/nativeassets/netstandard2.0/"
    "./packages/Microsoft.ML/runtimes/osx-x64/native"
    "./packages/Microsoft.ML.FastTree/runtimes/osx-x64/native"
    ]
    

let dependencies =
    //match RuntimeInformation.OSArchitecture with  | Architecture.X86   | Architecture.X64
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

    //| a -> failwithf "%A : Architecture not supported." a

dependencies 
|> Seq.iter (fun dep -> 
    let path = Path.Combine(__SOURCE_DIRECTORY__,dep)
    Environment.SetEnvironmentVariable("Path",
        Environment.GetEnvironmentVariable("Path") + ";" + path)
    )    


// Reference .dlls
// Note: referencing by path is needed also when referencing the directory.
#r "./packages/Newtonsoft.Json/lib/netstandard2.0/Newtonsoft.Json.dll"

#r "./packages/Microsoft.ML.DataView/lib/netstandard2.0/Microsoft.ML.DataView.dll"
#r "./packages/System.Memory/lib/netstandard2.0/System.Memory.dll"
#r "./packages/System.Runtime.CompilerServices.Unsafe/lib/netstandard2.0/System.Runtime.CompilerServices.Unsafe.dll"
#r "./packages/System.Numerics.Vectors/lib/netstandard2.0/System.Numerics.Vectors.dll"
#r "./packages/Microsoft.ML.CpuMath/lib/netstandard2.0/Microsoft.ML.CpuMath.dll"
#r "./packages/System.Threading.Tasks.Dataflow/lib/netstandard2.0/System.Threading.Tasks.Dataflow.dll"

#r "./packages/System.Collections.Immutable/lib/netstandard2.0/System.Collections.Immutable.dll"

#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Core.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Data.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.StandardTrainers.dll"
#r "./packages/Microsoft.ML.FastTree/lib/netstandard2.0/Microsoft.ML.FastTree.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.KMeansClustering.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.PCA.dll"
#r "./packages/Microsoft.ML/lib/netstandard2.0/Microsoft.ML.Transforms.dll"

#r "./src/FSharpML/bin/Release/netstandard2.1/FSharpML.dll"

