plugins {
    id("java-library")
}

group = "dev.simplified"
version = "1.0.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

repositories {
    mavenCentral()
    maven(url = "https://jitpack.io")
}

dependencies {
    // Simplified Libraries
    api("com.github.simplified-dev:collections") { version { strictly("a5f41e0") } }
    api("com.github.simplified-dev:utils") { version { strictly("5c6c96a") } }

    // JetBrains Annotations
    api(libs.annotations)

    // Logging
    api(libs.log4j2.api)

    // Lombok Annotations
    compileOnly(libs.lombok)
    annotationProcessor(libs.lombok)
    testCompileOnly(libs.lombok)
    testAnnotationProcessor(libs.lombok)

    // Tests
    testImplementation(libs.hamcrest)
    testImplementation(libs.junit.jupiter.api)
    testRuntimeOnly(libs.junit.jupiter.engine)
    testImplementation(libs.junit.platform.launcher)
}

// The Vector API lives in the jdk.incubator.vector module. It is referenced only from the
// package-private VectorOps class, which is loaded lazily by PixelVector when the module is
// resolvable at runtime. Consumers who don't pass --add-modules jdk.incubator.vector to their
// JVM transparently get the scalar path - see PixelBuffer class javadoc.
val vectorModuleArgs = listOf("--add-modules", "jdk.incubator.vector")

tasks.withType<JavaCompile> {
    options.compilerArgs.addAll(vectorModuleArgs)
    options.compilerArgs.add("-Xlint:-options")
}

tasks.test {
    useJUnitPlatform()
    jvmArgs(vectorModuleArgs)
    // Forward the libwebp-parity harness's Python-interpreter override to the
    // test JVM so golden-reference tests can locate a Python install with the
    // `webp` package when the default one on PATH doesn't have it. See
    // `ConformanceHelper.startPython` for usage.
    System.getProperty("vp8.pythonBin")?.let { systemProperty("vp8.pythonBin", it) }
}
