plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

val encodedSigningStore = rootProject.file("signing/muscriptor-dev.p12.b64")
val persistentSigningStore = layout.buildDirectory.file("signing/muscriptor-dev.p12").get().asFile
check(encodedSigningStore.isFile) { "Persistent sideload signing store is missing" }
persistentSigningStore.parentFile.mkdirs()
persistentSigningStore.writeBytes(
    java.util.Base64.getMimeDecoder().decode(encodedSigningStore.readText().trim()),
)

android {
    namespace = "dev.cardrhyme.muscriptormobile"
    compileSdk = 36

    signingConfigs {
        create("persistentSideload") {
            storeFile = persistentSigningStore
            storePassword = "muscriptor-dev"
            keyAlias = "muscriptor"
            keyPassword = "muscriptor-dev"
            storeType = "PKCS12"
        }
    }

    defaultConfig {
        applicationId = "dev.cardrhyme.muscriptormobile"
        minSdk = 26
        targetSdk = 35
        versionCode = 10
        versionName = "0.6.0"

        ndk {
            abiFilters += "arm64-v8a"
        }

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        debug {
            signingConfig = signingConfigs.getByName("persistentSideload")
        }
        release {
            isMinifyEnabled = false
            signingConfig = signingConfigs.getByName("persistentSideload")
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }

    packaging {
        jniLibs {
            useLegacyPackaging = true
            pickFirsts += setOf("**/libc++_shared.so")
        }
        resources {
            excludes += setOf("META-INF/DEPENDENCIES", "META-INF/LICENSE*", "META-INF/NOTICE*")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.16.0")
    implementation("androidx.activity:activity-ktx:1.10.1")
    implementation("androidx.appcompat:appcompat:1.7.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.9.1")
    implementation("androidx.media3:media3-exoplayer:1.10.1")
    implementation("com.google.android.material:material:1.12.0")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.10.2")
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.26.0")
    implementation("com.github.wendykierp:JTransforms:3.1")

    testImplementation("junit:junit:4.13.2")
}
