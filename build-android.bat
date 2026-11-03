@echo off
rem Construit l'app Android (lecteur) ; "build-android.bat install" l'installe sur le téléphone branché (adb).
rem Prérequis : Android Studio (SDK + JBR), NDK r30, Rust + cible aarch64-linux-android, cargo-ndk.
rem Les poids du modèle ne sont pas dans l'APK : l'app les télécharge au premier lancement.
setlocal
cd /d "%~dp0"
set "ANDROID_NDK_HOME=%LOCALAPPDATA%\Android\Sdk\ndk\30.0.16248370"
if "%JAVA_HOME%"=="" set "JAVA_HOME=C:\Program Files\Android\Android Studio\jbr"
set "PATH=%USERPROFILE%\.cargo\bin;%PATH%"

cargo ndk -t arm64-v8a -P 29 -o android\app\src\main\jniLibs build --release -p vocalcode-android || exit /b 1

pushd android
call gradlew.bat assembleRelease --console=plain || exit /b 1
popd
echo.
echo APK : android\app\build\outputs\apk\release\app-release.apk
if "%1"=="install" adb install -r android\app\build\outputs\apk\release\app-release.apk
