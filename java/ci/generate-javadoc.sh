#!/bin/bash
set -ex

# Set workspace
WORKSPACE=`pwd`
OUT_PATH="$WORKSPACE/javadoc_output"
rm -rf "$OUT_PATH"
mkdir -p "$OUT_PATH"

# Check and install JDK17
JDK_PATH="/usr/lib/jvm/java-17-openjdk"
REQUIRED_JDK_VERSION="17"

# Check if JDK is installed at expected path
if [ -d "$JDK_PATH" ]; then
    echo "Found JDK 17 at $JDK_PATH"
    export JAVA_HOME="$JDK_PATH"
    export PATH="$JAVA_HOME/bin:$PATH"
else
    echo "JDK 17 not found at $JDK_PATH, installing..."
    sudo yum install -y java-17-openjdk-devel
    
    # Verify installation
    if [ -d "$JDK_PATH" ]; then
        echo "JDK 17 installed successfully at $JDK_PATH"
        export JAVA_HOME="$JDK_PATH"
        export PATH="$JAVA_HOME/bin:$PATH"
    else
        echo "ERROR: JDK 17 installation failed. Expected path $JDK_PATH not found."
        exit 1
    fi
fi

# Verify Java version
echo "Using Java at: $JAVA_HOME/bin/java"
java -version
INSTALLED_VERSION=$(java -version 2>&1 | head -1 | cut -d'"' -f2 | cut -d'.' -f1)
if [ "$INSTALLED_VERSION" != "$REQUIRED_JDK_VERSION" ]; then
    echo "ERROR: Java version $INSTALLED_VERSION found, but $REQUIRED_JDK_VERSION required"
    exit 1
fi

pushd "$WORKSPACE/java"

# Get classpath using Maven
TEMP_CP_FILE=$(mktemp)
trap 'rm -f "$TEMP_CP_FILE"' EXIT
mvn -B dependency:build-classpath -Dmdep.outputFile="$TEMP_CP_FILE"
CLASSPATH=$(cat "$TEMP_CP_FILE")

# Generate Javadoc
"$JAVA_HOME/bin/javadoc" \
  -d "$OUT_PATH" \
  -sourcepath "src/main/java" \
  -subpackages "ai.rapids.cudf" \
  -classpath "$CLASSPATH" \
  --release 17 \
  -encoding UTF-8 \
  -charset UTF-8 \
  -docencoding UTF-8 \
  -quiet \
  -notimestamp \
  -Xdoclint:none

echo "Successfully generated Javadoc"

# Wrap static html files into war
jar cf cudf-javadoc.war -C $OUT_PATH .
