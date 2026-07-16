name := "nako-ribs-ssm"
version := "0.1.0"
scalaVersion := "3.3.1"

resolvers ++= Resolver.sonatypeOssRepos("releases")

libraryDependencies ++= Seq(
  "ch.unibas.cs.gravis" %% "scalismo" % "1.0-RC1"
  // No scalismo-ui / scalismo-vtk: 1.0-RC1 dropped VTK from the core,
  // so there are no native deps to pull in.
)

// Fork JVM so Scalismo's native libraries load correctly.
fork := true

// Forward forked stdio unmodified instead of routing it through sbt's
// logger.  Default behaviour wraps stderr lines in `[error] …`, which
// makes every INFO-level log line from RibRegistration look like an
// error.  StdoutOutput passes through both streams with no prefix.
run / outputStrategy := Some(StdoutOutput)

// Java's `TimeZone.getDefault()` can fall back to UTC on Linux nodes
// where the JVM doesn't pick up the host's IANA timezone from glibc
// Detect the host zone from `/etc/localtime` (POSIX symlink) and pass
// it explicitly to the forked JVM so its log timestamps match the 
// Python pipeline driver's.
javaOptions += {
  val tz = Option(System.getenv("TZ")).filter(_.nonEmpty)
    .orElse {
      val p = java.nio.file.Paths.get("/etc/localtime")
      scala.util.Try(p.toRealPath().toString).toOption.flatMap { s =>
        val idx = s.lastIndexOf("/zoneinfo/")
        if (idx >= 0) Some(s.substring(idx + "/zoneinfo/".length)) else None
      }
    }
    .getOrElse(java.util.TimeZone.getDefault.getID)
  s"-Duser.timezone=$tz"
}
