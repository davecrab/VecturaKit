// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
  name: "VecturaKit",
  platforms: [
    .macOS(.v14),
    .iOS(.v17),
    .tvOS(.v17),
    .visionOS(.v1),
    .watchOS(.v10),
  ],
  products: [
    .library(
      name: "VecturaKit",
      targets: ["VecturaKit"]
    ),
    .library(
      name: "VecturaMLXKit",
      targets: ["VecturaMLXKit"]
    ),
    .library(
      name: "VecturaExternalKit",
      targets: ["VecturaExternalKit"]
    ),
    .library(
      name: "VecturaCore",
      targets: ["VecturaCore"]
    ),
    .executable(
      name: "vectura-cli",
      targets: ["VecturaCLI"]
    ),
    .executable(
      name: "vectura-mlx-cli",
      targets: ["VecturaMLXCLI"]
    ),
    .executable(
      name: "vectura-external-cli",
      targets: ["VecturaExternalCLI"]
    ),
  ],
  dependencies: [
    .package(url: "https://github.com/jkrukowski/swift-embeddings.git", from: "0.0.10"),
    .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.4.0"),
    .package(url: "https://github.com/davecrab/mlx-swift-examples", branch: "main"),
  ],
  targets: [
    .target(
      name: "VecturaCore",
      dependencies: [],
      cSettings: [
        .define("ACCELERATE_NEW_LAPACK"),
        .define("ACCELERATE_LAPACK_ILP64"),
      ]
    ),
    .target(
      name: "VecturaKit",
      dependencies: [
        "VecturaCore",
        .product(name: "Embeddings", package: "swift-embeddings")
      ]
    ),
    .target(
      name: "VecturaMLXKit",
      dependencies: [
        "VecturaCore",
        .product(name: "MLXEmbedders", package: "mlx-swift-examples"),
      ]
    ),
    .target(
      name: "VecturaExternalKit",
      dependencies: ["VecturaCore"]
    ),
    .executableTarget(
      name: "VecturaCLI",
      dependencies: [
        "VecturaKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .executableTarget(
      name: "VecturaMLXCLI",
      dependencies: [
        "VecturaMLXKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .executableTarget(
      name: "VecturaExternalCLI",
      dependencies: [
        "VecturaExternalKit",
        .product(name: "ArgumentParser", package: "swift-argument-parser"),
      ]
    ),
    .testTarget(
      name: "VecturaKitTests",
      dependencies: ["VecturaKit"]
    ),
    .testTarget(
      name: "VecturaMLXKitTests",
      dependencies: ["VecturaMLXKit"]
    ),
    .testTarget(
      name: "VecturaExternalKitTests",
      dependencies: ["VecturaExternalKit"]
    ),
  ]
)
