package nako.ribs

import scalismo.geometry.*
import scalismo.mesh.*
import scalismo.io.MeshIO

import java.io.File
import java.nio.file.Files


// ────────────────────────────────────────────────────────────────────────────
// Centrality-based template-patient selection.
//
// Picks the most central template patient from a random subsample of the
// cohort.  For each of the 24 rib identities, computes pairwise
// MeshMetrics.avgDistance on the subsample; each patient accumulates a
// normalised centrality score across all rib identities, and the patient
// with the lowest aggregate score is written to ``--outTxt``.
// ────────────────────────────────────────────────────────────────────────────

object TemplateBuilder:
  /**
   * Pick the most central template patient from a subsample.
   *
   * For each of the 24 rib identities, compute pairwise ``avgDistance`` on
   * the subsample.  Each patient accumulates a normalised centrality score
   * across all rib identities.  The patient with the lowest aggregate score
   * is chosen — guaranteeing the template is consistently central, not just
   * for a few rib levels.
   *
   * Patients with incomplete rib sets (< 24 STLs) are excluded.
   */
  def main(args: Array[String]): Unit =
    val argMap = args.grouped(2).collect { case Array(k, v) => k -> v }.toMap

    val inputDir   = new File(argMap.getOrElse("--input",  sys.error("--input required")))
    val outTxtFile = new File(argMap.getOrElse("--outTxt", sys.error("--outTxt required")))
    val sampleSize = argMap.getOrElse("--sample", "50").toInt
    val seed       = argMap.getOrElse("--seed",   "42").toLong

    val ribLabels = argMap.getOrElse("--rib-labels", "40,41,42,43,44,45,46,47,48,49,50,51")
      .split(",").map(_.trim.toInt).toSeq
    val sides = Seq("L", "R")
    val rng   = new scala.util.Random(seed)

    // Discover all patient IDs from the sharded layout
    // ``<inputDir>/<block>/<pid>/`` and keep only those with a complete
    // 24-rib STL set.
    val pidDirs: Seq[File] =
      Option(inputDir.listFiles()).getOrElse(Array.empty[File]).toSeq
        .filter(_.isDirectory)
        .flatMap(blockDir =>
          Option(blockDir.listFiles()).getOrElse(Array.empty[File]).toSeq)
        .filter(d => d.isDirectory && d.getName.forall(_.isDigit))

    val expectedCount = ribLabels.length * sides.length  // 24
    val completePids: Seq[String] = pidDirs.flatMap { d =>
      val pid = d.getName
      val complete = ribLabels.forall { lab =>
        sides.forall { side =>
          new File(d, s"${pid}_rib${lab}_${side}.stl").exists()
        }
      }
      if complete then Some(pid) else None
    }.sorted

    println(s"[Template] ${completePids.length} patients with complete " +
            s"$expectedCount-rib sets (from ${pidDirs.size} total)")

    if completePids.isEmpty then
      println("[Template] No complete patients found; aborting.")
      sys.exit(1)

    val sampPids = rng.shuffle(completePids).take(math.min(sampleSize, completePids.length))
    println(s"[Template] Subsample: ${sampPids.length} patients")

    // For each rib identity, compute pairwise distances and accumulate
    // normalised centrality scores per patient.  Normalising per-rib by
    // dividing by (n-1) makes each rib contribute equally regardless of
    // absolute distance scale.
    val aggregateScores = scala.collection.mutable.Map[String, Double]()
      .withDefaultValue(0.0)
    var ribsProcessed = 0

    for label <- ribLabels; side <- sides do
      val ribId  = s"rib${label}_$side"
      val suffix = s"_${ribId}.stl"

      val meshes: Seq[(String, TriangleMesh[_3D])] = sampPids.flatMap { pid =>
        val f = new File(PatientPath.patientDir(inputDir, pid), s"${pid}$suffix")
        MeshIO.readMesh(f).toOption.map(pid -> _)
      }

      if meshes.length < 2 then
        println(s"[Template]   $ribId: only ${meshes.length} mesh(es) — skipping")
      else
        val n = meshes.length
        val scores = Array.fill(n)(0.0)
        // MeshMetrics.avgDistance is symmetric, so compute the upper triangle
        // only and accumulate into both endpoints (halves the pairwise calls).
        val t0 = System.nanoTime()
        var i = 0
        while i < n do
          var j = i + 1
          while j < n do
            val d = MeshMetrics.avgDistance(meshes(i)._2, meshes(j)._2)
            scores(i) += d
            scores(j) += d
            j += 1
          i += 1
        val elapsedSec = (System.nanoTime() - t0) / 1e9

        for idx <- scores.indices do
          val pid = meshes(idx)._1
          aggregateScores(pid) += scores(idx) / (n - 1)

        val bestForRib = scores.zipWithIndex.minBy(_._1)
        ribsProcessed += 1
        println(f"[Template]   $ribsProcessed%2d/$expectedCount $ribId  " +
                f"n=$n  best=${meshes(bestForRib._2)._1}  ${elapsedSec}%.1fs")

    val ranked = aggregateScores.toSeq.sortBy(_._2)
    println(s"\n[Template] Aggregate ranking (${ranked.length} patients, " +
            s"$ribsProcessed rib identities):")
    println("[Template] Top 5 candidates (lower = more central across all ribs):")
    for ((pid, score) <- ranked.take(5)) do
      println(f"  patient $pid  aggregate score = $score%.3f mm")

    val bestPid   = ranked.head._1
    val bestScore = ranked.head._2

    println(f"\n[Template] Chosen: patient $bestPid  " +
            f"(aggregate score = $bestScore%.3f mm)")
    outTxtFile.getParentFile.mkdirs()
    Files.writeString(outTxtFile.toPath, bestPid)
    println(s"[Template] Written to ${outTxtFile.getAbsolutePath}")
