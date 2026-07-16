package nako.ribs


/* ----------------------------------------------------------------------
 *  Per-patient rib registration entry point for the scalismo_ssm sbt project.
 *
 *  Pipeline per (patient, rib):
 *    1. Whole-cage Procrustes pre-alignment on all 24 rib centroids (rigid,
 *       bilateral-symmetry guard).
 *    2. PCA principal-axis alignment with brute-force 4-sign search.
 *    3. Similarity ICP (R + s + t, 30 iters, trimmed-mean 80%).
 *    4. Rigid ICP polish (25 iters, trimmed-mean 85%).
 *    5. Build per-rib Similarity3D S via Umeyama on final correspondences;
 *       bring target into reference frame via S^-1.
 *    6. Coarse-to-fine non-rigid ICP (4 passes) with distance + boundary
 *       + normal-cosine filters; GP rebuilt between passes with shrinking
 *       outer length-scale.
 *    7. Push the fitted reference forward through S, then through the inverse
 *       whole-cage transform, back into the patient's scanner frame.
 *
 *  Stage 1 runs parallel over patients; stage 2 is per-rib outer × per-patient
 *  inner parallelism (fresh thread pool per rib).
 *
 *  Run with:
 *
 *    cd environment/scalismo_ssm
 *    ./sbt_build.sh "runMain nako.ribs.RibRegistration \
 *        --input <data/extracted_stl> --output <out_dir> \
 *        --template-pid <PID> --workers 4"
 *
 *  Or via scripts/run_pipeline.py.
 * ---------------------------------------------------------------------- */

import scalismo.geometry.{_3D, Point, Point3D, EuclideanVector, EuclideanVector3D, Landmark}
import scalismo.mesh.{TriangleMesh, TriangleMesh3D}
import scalismo.mesh.boundingSpheres.{ClosestPointInTriangle, ClosestPointOnLine, ClosestPointIsVertex}
import scalismo.io.MeshIO
import scalismo.common.{PointId, Field, RealSpace, Domain, EuclideanSpace3D}
import scalismo.common.interpolation.TriangleMeshInterpolator3D
import scalismo.kernels.{DiagonalKernel3D, GaussianKernel3D}
import scalismo.statisticalmodel.{GaussianProcess, LowRankGaussianProcess, MultivariateNormalDistribution}
import scalismo.numerics.UniformMeshSampler3D
import scalismo.registration.LandmarkRegistration
import scalismo.transformations.Transformation
import scalismo.utils.Random

import breeze.linalg.{DenseMatrix, DenseVector, eigSym, svd, det}

import java.io.File
import java.util.concurrent.{Executors, TimeUnit}
import java.util.concurrent.atomic.AtomicInteger
import scala.concurrent.{Await, ExecutionContext, Future}
import scala.concurrent.duration.Duration
import scala.util.{Try, Success, Failure}


// ────────────────────────────────────────────────────────────────────────────
// Formatting helper
// ────────────────────────────────────────────────────────────────────────────

object FmtTime:
  def apply(ms: Long): String =
    val s = ms / 1000
    if s < 60 then f"${s}s"
    else if s < 3600 then f"${s / 60}m${s % 60}s"
    else f"${s / 3600}h${(s % 3600) / 60}m"


// ────────────────────────────────────────────────────────────────────────────
// Logging. Format matches src/utils/logging.py
// ("HH:MM:SS  LEVEL(8)  NAME(28.28)  MESSAGE") so Scala lines interleave
// with Python-side logs under run_pipeline.py. Writes go to stderr,
// serialised on a mutex shared with ProgressBar so log lines don't tear
// a live bar.
// ────────────────────────────────────────────────────────────────────────────

private object StderrMutex

object Log:
  private val nameField = "RibRegistration".padTo(28, ' ').take(28)
  private val tsFmt     = java.time.format.DateTimeFormatter.ofPattern("HH:mm:ss")

  private[ribs] def emit(level: String, msg: String): Unit = StderrMutex.synchronized {
    val ts  = java.time.LocalTime.now().format(tsFmt)
    val lvl = level.padTo(8, ' ')
    System.err.println(s"$ts  $lvl  $nameField  $msg")
  }
  def info(msg:  String): Unit = emit("INFO",     msg)
  def warn(msg:  String): Unit = emit("WARNING",  msg)
  def error(msg: String): Unit = emit("ERROR",    msg)


/** Thread-safe single-line progress bar in tqdm format. Redraws at most
 *  every `minIntervalMs`. `tick()` is concurrent-safe; `log(level, msg)`
 *  emits a log line above the bar without tearing. */
final class ProgressBar(desc: String, val total: Int, minIntervalMs: Long = 250L):
  private val n           = new java.util.concurrent.atomic.AtomicInteger(0)
  private val startMs     = System.currentTimeMillis()
  private var lastDrawMs  = 0L
  private var lastLineLen = 0

  private def render(curN: Int): String =
    val frac      = if total > 0 then curN.toDouble / total else 0.0
    val pct       = (frac * 100).toInt
    val barLen    = 10
    val filled    = math.min(barLen, math.max(0, (frac * barLen).toInt))
    val barStr    = ("█" * filled) + (" " * (barLen - filled))
    val elapsedMs = System.currentTimeMillis() - startMs
    val rate      = curN / math.max(0.001, elapsedMs / 1000.0)
    val eta       = if rate > 0 && curN < total then ((total - curN) / rate * 1000.0).toLong else 0L
    f"$desc: $pct%3d%%|$barStr| $curN/$total [${FmtTime(elapsedMs)}<${FmtTime(eta)}, $rate%.2f it/s]"

  /** Increment the counter; redraw if `minIntervalMs` elapsed or total reached. */
  def tick(): Unit =
    val cur = n.incrementAndGet()
    val now = System.currentTimeMillis()
    StderrMutex.synchronized {
      if cur >= total || now - lastDrawMs >= minIntervalMs then
        drawLocked(cur, now)
    }

  /** Final redraw + newline (always emits, ignoring throttle). */
  def finish(): Unit = StderrMutex.synchronized {
    drawLocked(n.get(), System.currentTimeMillis())
    System.err.println()
    lastLineLen = 0
  }

  /** Emit a log line above the bar, then redraw the bar. */
  def log(level: String, msg: String): Unit = StderrMutex.synchronized {
    if lastLineLen > 0 then
      System.err.print("\r" + (" " * lastLineLen) + "\r")
      lastLineLen = 0
    Log.emit(level, msg)
    val cur = n.get()
    drawLocked(cur, System.currentTimeMillis())
  }

  private def drawLocked(cur: Int, now: Long): Unit =
    val line   = render(cur)
    val padded = line + (" " * math.max(0, lastLineLen - line.length))
    System.err.print("\r" + padded)
    System.err.flush()
    lastLineLen = line.length
    lastDrawMs  = now


// ────────────────────────────────────────────────────────────────────────────
// Per-patient sharded directory layout: <root>/<block>/<pid>/, block = pid/1000.
// ────────────────────────────────────────────────────────────────────────────

object PatientPath:
  def patientDir(root: File, pid: String): File =
    new File(new File(root, (pid.toLong / 1000).toString), pid)


// ────────────────────────────────────────────────────────────────────────────
// Similarity3D: p' = rotCenter + scale * R * (p - rotCenter) + translation.
// Forward maps reference → target frame; inverse maps target → reference.
// ────────────────────────────────────────────────────────────────────────────

final case class Similarity3D(
    rotation: DenseMatrix[Double],     // 3x3 orthonormal, det = +1
    scale: Double,                     // > 0
    rotCenter: Point[_3D],
    translation: EuclideanVector[_3D],
):
  def apply(p: Point[_3D]): Point[_3D] =
    val dx = p.x - rotCenter.x
    val dy = p.y - rotCenter.y
    val dz = p.z - rotCenter.z
    val rx = rotation(0, 0) * dx + rotation(0, 1) * dy + rotation(0, 2) * dz
    val ry = rotation(1, 0) * dx + rotation(1, 1) * dy + rotation(1, 2) * dz
    val rz = rotation(2, 0) * dx + rotation(2, 1) * dy + rotation(2, 2) * dz
    Point3D(
      rotCenter.x + scale * rx + translation.x,
      rotCenter.y + scale * ry + translation.y,
      rotCenter.z + scale * rz + translation.z,
    )

  def inverse: Similarity3D =
    val invRot = rotation.t
    val invScale = 1.0 / scale
    val tVec = DenseVector(translation.x, translation.y, translation.z)
    val invT = (invRot * tVec) * (-invScale)
    Similarity3D(
      invRot,
      invScale,
      rotCenter,
      EuclideanVector3D(invT(0), invT(1), invT(2)),
    )


// ────────────────────────────────────────────────────────────────────────────
// Whole-cage rigid Procrustes alignment via rib centroids.
// ────────────────────────────────────────────────────────────────────────────

object WholeCageAlignment:

  def meshCentroid(mesh: TriangleMesh[_3D]): EuclideanVector[_3D] =
    val pts = mesh.pointSet.points.toIndexedSeq
    val n = pts.length
    pts.foldLeft(EuclideanVector.zeros[_3D]) { (acc, p) =>
      acc + p.toVector
    } * (1.0 / n)

  def computeRibCentroids(
      inputDir: File,
      pid: String,
      ribLabels: Seq[Int],
      sides: Seq[String],
  ): Map[String, EuclideanVector[_3D]] =
    val centroids = scala.collection.mutable.Map.empty[String, EuclideanVector[_3D]]
    val pdir = PatientPath.patientDir(inputDir, pid)
    for label <- ribLabels; side <- sides do
      val ribId = s"rib${label}_$side"
      val f = new File(pdir, s"${pid}_${ribId}.stl")
      if f.exists() then
        MeshIO.readMesh(f) match
          case Success(mesh) => centroids(ribId) = meshCentroid(mesh)
          case Failure(_)    => () // skip unreadable
    centroids.toMap

  /** Rigid Procrustes (R + t, no scaling) with bilateral-symmetry guard:
   *  trace < 1 (>90°) → identity rotation; for rib cages in scanner
   *  coordinates a large rotation is almost certainly an L/R flip. */
  def procrustes(
      srcPoints: DenseMatrix[Double],   // N x 3
      tgtPoints: DenseMatrix[Double],   // N x 3
  ): (DenseMatrix[Double], DenseVector[Double]) =
    val n = srcPoints.rows
    require(n == tgtPoints.rows && n >= 3, s"Need >=3 matched points, got $n")

    val srcMean = DenseVector.zeros[Double](3)
    val tgtMean = DenseVector.zeros[Double](3)
    var i = 0
    while i < n do
      for d <- 0 until 3 do
        srcMean(d) += srcPoints(i, d)
        tgtMean(d) += tgtPoints(i, d)
      i += 1
    srcMean :*= (1.0 / n)
    tgtMean :*= (1.0 / n)

    val srcCentered = srcPoints.copy
    val tgtCentered = tgtPoints.copy
    i = 0
    while i < n do
      for d <- 0 until 3 do
        srcCentered(i, d) -= srcMean(d)
        tgtCentered(i, d) -= tgtMean(d)
      i += 1

    val h = srcCentered.t * tgtCentered
    val svd.SVD(u, _, vt) = svd(h): @unchecked
    val v = vt.t
    val d = det(v * u.t)
    val sign = DenseMatrix.eye[Double](3)
    if d < 0 then sign(2, 2) = -1.0
    var rotation = v * sign * u.t

    val trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2)
    if trace < 1.0 then rotation = DenseMatrix.eye[Double](3)

    val translation = tgtMean - rotation * srcMean
    (rotation, translation)

  /** Umeyama 1991 similarity Procrustes: minimises Σ ‖tgt - (s·R·src + t)‖².
   *  Same bilateral-symmetry guard as [[procrustes]]. */
  def procrustesSimilarity(
      srcPoints: DenseMatrix[Double],
      tgtPoints: DenseMatrix[Double],
  ): (DenseMatrix[Double], Double, DenseVector[Double]) =
    val n = srcPoints.rows
    require(n == tgtPoints.rows && n >= 3, s"Need >=3 matched points, got $n")

    val srcMean = DenseVector.zeros[Double](3)
    val tgtMean = DenseVector.zeros[Double](3)
    var i = 0
    while i < n do
      for d <- 0 until 3 do
        srcMean(d) += srcPoints(i, d)
        tgtMean(d) += tgtPoints(i, d)
      i += 1
    srcMean :*= (1.0 / n)
    tgtMean :*= (1.0 / n)

    val srcCentered = srcPoints.copy
    val tgtCentered = tgtPoints.copy
    i = 0
    while i < n do
      for d <- 0 until 3 do
        srcCentered(i, d) -= srcMean(d)
        tgtCentered(i, d) -= tgtMean(d)
      i += 1

    val h = srcCentered.t * tgtCentered
    val svd.SVD(u, sigma, vt) = svd(h): @unchecked
    val v = vt.t
    val d = det(v * u.t)
    val signMat = DenseMatrix.eye[Double](3)
    if d < 0 then signMat(2, 2) = -1.0
    var rotation = v * signMat * u.t

    val trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2)
    if trace < 1.0 then rotation = DenseMatrix.eye[Double](3)

    var sumSqSrc = 0.0
    i = 0
    while i < n do
      val x = srcCentered(i, 0); val y = srcCentered(i, 1); val z = srcCentered(i, 2)
      sumSqSrc += x * x + y * y + z * z
      i += 1
    val sigSum = sigma(0) + sigma(1) + (if d < 0 then -sigma(2) else sigma(2))
    val sc = if sumSqSrc > 1e-12 then sigSum / sumSqSrc else 1.0

    val translation = tgtMean - (rotation * srcMean) * sc
    (rotation, sc, translation)

  /** Returns Some((R, t)) if ≥6 rib centroids match between template and patient,
   *  None otherwise. */
  def alignToTemplate(
      templateCentroids: Map[String, EuclideanVector[_3D]],
      patientCentroids: Map[String, EuclideanVector[_3D]],
      orderedRibIds: Seq[String],
  ): Option[(DenseMatrix[Double], DenseVector[Double])] =
    val matched = orderedRibIds.filter { id =>
      templateCentroids.contains(id) && patientCentroids.contains(id)
    }
    if matched.length < 6 then return None
    val n = matched.length
    val srcPts = DenseMatrix.zeros[Double](n, 3)
    val tgtPts = DenseMatrix.zeros[Double](n, 3)
    for (id, idx) <- matched.zipWithIndex do
      val s = patientCentroids(id)
      val t = templateCentroids(id)
      srcPts(idx, 0) = s.x; srcPts(idx, 1) = s.y; srcPts(idx, 2) = s.z
      tgtPts(idx, 0) = t.x; tgtPts(idx, 1) = t.y; tgtPts(idx, 2) = t.z
    Some(procrustes(srcPts, tgtPts))

  def transformMesh(
      mesh: TriangleMesh[_3D],
      rotation: DenseMatrix[Double],
      translation: DenseVector[Double],
  ): TriangleMesh[_3D] =
    val pts = mesh.pointSet.points.toIndexedSeq
    val newPts = pts.map { p =>
      val v = DenseVector(p.x, p.y, p.z)
      val tv = rotation * v + translation
      Point3D(tv(0), tv(1), tv(2))
    }
    TriangleMesh3D(newPts, mesh.triangulation)


// ────────────────────────────────────────────────────────────────────────────
// Geometry helpers
// ────────────────────────────────────────────────────────────────────────────

object Geom:

  def boundingBoxDiagonal(mesh: TriangleMesh[_3D]): Double =
    val pts = mesh.pointSet.points.toIndexedSeq
    val xs = pts.map(_.x); val ys = pts.map(_.y); val zs = pts.map(_.z)
    val dx = xs.max - xs.min
    val dy = ys.max - ys.min
    val dz = zs.max - zs.min
    math.sqrt(dx * dx + dy * dy + dz * dz)

  def centroid(mesh: TriangleMesh[_3D]): Point[_3D] =
    val pts = mesh.pointSet.points.toIndexedSeq
    val n = pts.size
    Point3D(
      pts.iterator.map(_.x).sum / n,
      pts.iterator.map(_.y).sum / n,
      pts.iterator.map(_.z).sum / n,
    )

  /** Centroid + 3×3 axes (columns = eigenvectors of vertex covariance,
   *  descending eigenvalue, det = +1) + eigenvalues. */
  def principalFrame(mesh: TriangleMesh[_3D])
      : (Point[_3D], DenseMatrix[Double], DenseVector[Double]) =
    val pts = mesh.pointSet.points.toIndexedSeq
    val n = pts.size
    val c = centroid(mesh)
    val cov = DenseMatrix.zeros[Double](3, 3)
    var i = 0
    while i < n do
      val dx = pts(i).x - c.x
      val dy = pts(i).y - c.y
      val dz = pts(i).z - c.z
      cov(0, 0) += dx * dx; cov(0, 1) += dx * dy; cov(0, 2) += dx * dz
      cov(1, 1) += dy * dy; cov(1, 2) += dy * dz
      cov(2, 2) += dz * dz
      i += 1
    cov(1, 0) = cov(0, 1); cov(2, 0) = cov(0, 2); cov(2, 1) = cov(1, 2)
    cov :*= (1.0 / n)

    val es = eigSym(cov)
    // Reorder to descending eigenvalue.
    val order = Array(2, 1, 0)
    val evals = DenseVector(order.map(es.eigenvalues(_)))
    val evecs = DenseMatrix.zeros[Double](3, 3)
    for (k, col) <- order.zipWithIndex do
      evecs(::, col) := es.eigenvectors(::, k)

    if det(evecs) < 0 then evecs(::, 2) := evecs(::, 2) * (-1.0)
    (c, evecs, evals)

  /** Mean closest-point distance from a sample of source vertices to the target. */
  def meanSurfaceDistance(
      m: TriangleMesh[_3D],
      t: TriangleMesh[_3D],
      nSamples: Int = 400,
  )(using rng: Random): Double =
    val sampler = UniformMeshSampler3D(m, nSamples)
    val ds = sampler.sample().map(_._1).map { p =>
      (t.operations.closestPointOnSurface(p).point - p).norm
    }
    ds.sum / ds.size


// ────────────────────────────────────────────────────────────────────────────
// PCA principal-axis alignment with anatomy-preserving sign disambiguation.
// ────────────────────────────────────────────────────────────────────────────

object PoseAlign:

  /** Align the source mesh onto the target's principal frame. PCA eigenvectors
   *  have arbitrary sign; we resolve the ambiguity by requiring each target
   *  axis to point in the same half-space as the corresponding source axis
   *  (positive dot product). This is well-defined because both meshes are
   *  already in a shared frame from whole-cage alignment, and it cannot
   *  produce end-for-end flips (so posterior remains posterior). PC3 is set
   *  as PC1 × PC2 to guarantee a right-handed orthonormal frame (det = +1). */
  def alignByPrincipalAxes(
      source: TriangleMesh[_3D],
      target: TriangleMesh[_3D],
  ): TriangleMesh[_3D] =
    val (cS, axesS, _) = Geom.principalFrame(source)
    val (cT, axesTRaw, _) = Geom.principalFrame(target)

    val a = axesTRaw.copy
    if (a(::, 0) dot axesS(::, 0)) < 0.0 then a(::, 0) :*= -1.0
    if (a(::, 1) dot axesS(::, 1)) < 0.0 then a(::, 1) :*= -1.0
    val pc1 = a(::, 0).copy
    val pc2 = a(::, 1).copy
    a(0, 2) = pc1(1) * pc2(2) - pc1(2) * pc2(1)
    a(1, 2) = pc1(2) * pc2(0) - pc1(0) * pc2(2)
    a(2, 2) = pc1(0) * pc2(1) - pc1(1) * pc2(0)

    val R = a * axesS.t
    val tVec = DenseVector(cT.x, cT.y, cT.z) - R * DenseVector(cS.x, cS.y, cS.z)
    WholeCageAlignment.transformMesh(source, R, tVec)


// ────────────────────────────────────────────────────────────────────────────
// Trimmed-mean similarity / rigid ICP.
// ────────────────────────────────────────────────────────────────────────────

object IcpAlign:

  /** Similarity ICP (R + s + t). Per iter: closest-point pairs, drop the worst
   *  (1 − keepFrac), call similarity3DLandmarkRegistration. */
  def similarityICP(
      source: TriangleMesh[_3D],
      target: TriangleMesh[_3D],
      iters: Int = 30,
      sampleN: Int = 1500,
      keepFrac: Double = 0.8,
  ): TriangleMesh[_3D] =
    val n = source.pointSet.numberOfPoints
    val step = math.max(1, n / sampleN)
    val ids = (0 until n by step).map(PointId.apply)

    @scala.annotation.tailrec
    def loop(m: TriangleMesh[_3D], it: Int): TriangleMesh[_3D] =
      if it == 0 then m
      else
        val pairs = ids.map { id =>
          val p = m.pointSet.point(id)
          val q = target.operations.closestPointOnSurface(p).point
          (p, q)
        }
        val sorted = pairs.sortBy { case (a, b) => (a - b).norm }
        val kept = sorted.take((sorted.size * keepFrac).toInt)
        val t = LandmarkRegistration.similarity3DLandmarkRegistration(
          kept, center = Point3D(0, 0, 0))
        loop(m.transform(t), it - 1)

    loop(source, iters)

  /** Rigid ICP polish (R + t only). */
  def rigidICP(
      source: TriangleMesh[_3D],
      target: TriangleMesh[_3D],
      iters: Int = 25,
      sampleN: Int = 1500,
      keepFrac: Double = 0.85,
  ): TriangleMesh[_3D] =
    val n = source.pointSet.numberOfPoints
    val step = math.max(1, n / sampleN)
    val ids = (0 until n by step).map(PointId.apply)

    @scala.annotation.tailrec
    def loop(m: TriangleMesh[_3D], it: Int): TriangleMesh[_3D] =
      if it == 0 then m
      else
        val pairs = ids.map { id =>
          val p = m.pointSet.point(id)
          val q = target.operations.closestPointOnSurface(p).point
          (p, q)
        }
        val sorted = pairs.sortBy { case (a, b) => (a - b).norm }
        val kept = sorted.take((sorted.size * keepFrac).toInt)
        val t = LandmarkRegistration.rigid3DLandmarkRegistration(
          kept, center = Point3D(0, 0, 0))
        loop(m.transform(t), it - 1)

    loop(source, iters)


// ────────────────────────────────────────────────────────────────────────────
// Gaussian-process prior + non-rigid ICP with three correspondence filters.
// ────────────────────────────────────────────────────────────────────────────

final case class NRICPPass(
    iters: Int,
    noiseVariance: Double,
    maxDistanceMm: Double,
    numSamples: Int,
    cosThreshold: Double,
)

object NonRigid:

  /** Sum of two Gaussian DiagonalKernels. Outer scale (≈ ¼ bbox-diag, min 30 mm)
   *  captures global bending; inner scale (25 mm) captures local thickness. */
  def buildRibGP(
      reference: TriangleMesh[_3D],
      outerScale: Double,
      relTol: Double = 0.05,
  ): LowRankGaussianProcess[_3D, EuclideanVector[_3D]] =
    val s1 = math.max(outerScale, 30.0); val v1 = s1 * 0.4
    val s2 = 25.0;                       val v2 = s2 * 0.2
    val k1 = DiagonalKernel3D(GaussianKernel3D(sigma = s1) * (v1 * v1), 3)
    val k2 = DiagonalKernel3D(GaussianKernel3D(sigma = s2) * (v2 * v2), 3)
    val zeroMean = Field(RealSpace[_3D], (_: Point[_3D]) => EuclideanVector.zeros[_3D])
    val gp = GaussianProcess(zeroMean, k1 + k2)
    LowRankGaussianProcess.approximateGPCholesky(
      reference,
      gp,
      relativeTolerance = relTol,
      interpolator = TriangleMeshInterpolator3D[EuclideanVector[_3D]](),
    )

  /** Lift a closest-point-on-surface result from `prev` back to the reference
   *  frame. `prev` shares topology with `referenceMesh`, so ids map 1:1.
   *  Returns `(refAnchor, srcVertexId)` — barycentric-interpolated anchor +
   *  the largest-weight vertex id (for the normal lookup). */
  private def liftToReference(
      cp: scalismo.mesh.boundingSpheres.ClosestPointWithType,
      prev: TriangleMesh[_3D],
      referenceMesh: TriangleMesh[_3D],
  ): (Point[_3D], PointId) = cp match
    case ClosestPointInTriangle(_, _, tid, bc) =>
      val tri = prev.triangulation.triangle(tid)
      val p0 = referenceMesh.pointSet.point(tri.ptId1).toVector
      val p1 = referenceMesh.pointSet.point(tri.ptId2).toVector
      val p2 = referenceMesh.pointSet.point(tri.ptId3).toVector
      val anchor = (p0 * bc.a + p1 * bc.b + p2 * bc.c).toPoint
      val srcId =
        if bc.a >= bc.b && bc.a >= bc.c then tri.ptId1
        else if bc.b >= bc.c then tri.ptId2
        else tri.ptId3
      (anchor, srcId)
    case ClosestPointOnLine(_, _, (id0, id1), bc) =>
      val p0 = referenceMesh.pointSet.point(id0).toVector
      val p1 = referenceMesh.pointSet.point(id1).toVector
      val anchor = (p0 * (1.0 - bc) + p1 * bc).toPoint
      val srcId = if bc <= 0.5 then id0 else id1
      (anchor, srcId)
    case ClosestPointIsVertex(_, _, pid) =>
      (referenceMesh.pointSet.point(pid), pid)
    case other =>
      // Tetrahedron-case variants of ClosestPointWithType are unreachable for
      // a TriangleMesh; snap to the nearest vertex on prev as a fallback.
      val pid = prev.pointSet.findClosestPoint(other.point).id
      (referenceMesh.pointSet.point(pid), pid)

  /** Non-rigid ICP pass with distance + boundary + normal-cosine filters.
   *  GP is conditioned on observations anchored at the *original* reference;
   *  the deformation field is the posterior mean applied back to the
   *  reference, so deformations don't compose between iters.
   *
   *  When `bidirectional` is true, each iteration also samples the target,
   *  projects onto the current deformed reference, lifts back via barycentric
   *  coordinates, and feeds `(refAnchor, t − refAnchor, noise)` triples into
   *  the same posterior. Reverse observations carry only distance + normal
   *  filters (no boundary filter on the reverse side).
   *
   *  Returns `(fittedMesh, nFwd, nRev)`; `nRev` is 0 when bidirectional is off. */
  def nonRigidICP(
      referenceMesh: TriangleMesh[_3D],
      currentRef: TriangleMesh[_3D],
      targetInRef: TriangleMesh[_3D],
      gp: LowRankGaussianProcess[_3D, EuclideanVector[_3D]],
      pass: NRICPPass,
      bidirectional: Boolean = false,
  )(using rng: Random): (TriangleMesh[_3D], Int, Int) =
    val sampler = UniformMeshSampler3D(referenceMesh, pass.numSamples)
    val sampleIds = sampler.sample().map(_._1)
      .map(p => referenceMesh.pointSet.findClosestPoint(p).id)
      .distinct

    // Reverse-side target sample (fixed across iterations of this pass).
    // Empty when bidirectional is off so forward-sampler RNG state matches.
    val targetSamples: IndexedSeq[Point[_3D]] =
      if bidirectional then
        UniformMeshSampler3D(targetInRef, pass.numSamples).sample().map(_._1).toIndexedSeq
      else IndexedSeq.empty

    val targetNormals = targetInRef.vertexNormals.pointData
    val noise = MultivariateNormalDistribution(
      DenseVector.zeros[Double](3),
      DenseMatrix.eye[Double](3) * pass.noiseVariance,
    )

    @scala.annotation.tailrec
    def loop(prev: TriangleMesh[_3D], it: Int, lastFwd: Int, lastRev: Int): (TriangleMesh[_3D], Int, Int) =
      if it == 0 then (prev, lastFwd, lastRev)
      else
        val currentNormals = prev.vertexNormals.pointData
        val obs = sampleIds.flatMap { id =>
          val srcCur = prev.pointSet.point(id)
          val cp = targetInRef.operations.closestPointOnSurface(srcCur)
          val dist = (cp.point - srcCur).norm
          if dist > pass.maxDistanceMm then None
          else
            val tgtId = targetInRef.pointSet.findClosestPoint(cp.point).id
            if targetInRef.operations.pointIsOnBoundary(tgtId) then None
            else
              val nSrc = currentNormals(id.id)
              val nTgt = targetNormals(tgtId.id)
              if nSrc.dot(nTgt) < pass.cosThreshold then None
              else
                val refPt = referenceMesh.pointSet.point(id)
                Some((refPt, cp.point - refPt, noise))
        }

        val revObs =
          if !bidirectional then IndexedSeq.empty
          else targetSamples.flatMap { t =>
            val cp = prev.operations.closestPointOnSurface(t)
            val dist = (cp.point - t).norm
            if dist > pass.maxDistanceMm then None
            else
              val (refAnchor, srcId) = liftToReference(cp, prev, referenceMesh)
              val tgtVid = targetInRef.pointSet.findClosestPoint(t).id
              val nSrc = currentNormals(srcId.id)
              val nTgt = targetNormals(tgtVid.id)
              if nSrc.dot(nTgt) < pass.cosThreshold then None
              else Some((refAnchor, t - refAnchor, noise))
          }

        val combined = obs ++ revObs
        if combined.size < 10 then
          // Too few correspondences for a stable posterior; bail out of the pass.
          (prev, obs.size, revObs.size)
        else
          val posterior = gp.posterior(combined.toIndexedSeq)
          val nextMesh = referenceMesh.transform(p => p + posterior.mean(p))
          loop(nextMesh, it - 1, obs.size, revObs.size)

    loop(currentRef, pass.iters, 0, 0)

  /** Coarse-to-fine schedule (4 passes) with shrinking outer length-scale
   *  and tightening distance / normal gates. */
  def coarseToFineNonRigidICP(
      referenceMesh: TriangleMesh[_3D],
      targetInRef: TriangleMesh[_3D],
      initialOuterScale: Double,
      bidirectional: Boolean = false,
  )(using rng: Random): TriangleMesh[_3D] =
    val bbDiag = Geom.boundingBoxDiagonal(referenceMesh)
    val schedule = Seq(
      NRICPPass(iters = 15, noiseVariance = 5.0,
                maxDistanceMm = math.max(bbDiag * 0.10, 30.0),
                numSamples = 1500, cosThreshold = 0.3),
      NRICPPass(iters = 20, noiseVariance = 1.0,
                maxDistanceMm = math.max(bbDiag * 0.05, 15.0),
                numSamples = 2500, cosThreshold = 0.5),
      NRICPPass(iters = 25, noiseVariance = 0.25,
                maxDistanceMm = math.max(bbDiag * 0.02, 8.0),
                numSamples = 4000, cosThreshold = 0.6),
      NRICPPass(iters = 30, noiseVariance = 0.05,
                maxDistanceMm = math.max(bbDiag * 0.01, 4.0),
                numSamples = 6000, cosThreshold = 0.7),
    )
    schedule.zipWithIndex.foldLeft(referenceMesh) { case (mesh, (p, k)) =>
      val gpK = buildRibGP(mesh, outerScale = initialOuterScale * math.pow(0.8, k))
      val (out, _, _) = nonRigidICP(referenceMesh, mesh, targetInRef, gpK, p, bidirectional)
      out
    }


// ────────────────────────────────────────────────────────────────────────────
// Per-rib registration glue.
// ────────────────────────────────────────────────────────────────────────────

object Register:

  /** Outputs of one per-rib registration: the final fitted mesh plus two
   *  intermediates used by the methodology figure. All three meshes are in
   *  the cage-aligned target frame. */
  final case class RegisterResult(
      fitted: TriangleMesh[_3D],          // = gpFit; the mesh used downstream
      perRib: TriangleMesh[_3D],          // template after PCA + similarity ICP + rigid ICP
      gpFit:  TriangleMesh[_3D],          // template after non-rigid coarse-to-fine ICP
  )

  /** PCA → similarity ICP → rigid ICP → Umeyama-built Similarity3D S →
   *  target brought into reference frame via S^-1 → 4-pass non-rigid ICP →
   *  fitted reference pushed forward through S into the cage-aligned target
   *  frame. The driver applies the inverse whole-cage transform to reach
   *  the patient's scanner frame. */
  def registerOne(
      reference: TriangleMesh[_3D],
      targetCage: TriangleMesh[_3D],
      bidirectional: Boolean = false,
  )(using rng: Random): RegisterResult =

    val srcAfterPCA = PoseAlign.alignByPrincipalAxes(reference, targetCage)
    val srcAfterSim = IcpAlign.similarityICP(srcAfterPCA, targetCage,
                                             iters = 30, sampleN = 1500, keepFrac = 0.8)
    val srcAfterRigid = IcpAlign.rigidICP(srcAfterSim, targetCage,
                                          iters = 25, sampleN = 1500, keepFrac = 0.85)

    // Umeyama similarity fit on final correspondences: S maps ref → target frame.
    val nRef = reference.pointSet.numberOfPoints
    val sampleStep = math.max(1, nRef / 2000)
    val pairIds = (0 until nRef by sampleStep).map(PointId.apply)
    val rawPairs = pairIds.flatMap { id =>
      val refPt = reference.pointSet.point(id)
      val srcPt = srcAfterRigid.pointSet.point(id)
      val tgtPt = targetCage.operations.closestPointOnSurface(srcPt).point
      val dist = (tgtPt - srcPt).norm
      if dist < 10.0 then Some((refPt, tgtPt)) else None
    }
    if rawPairs.size < 50 then
      throw new RuntimeException(
        s"registerOne: too few similarity-fit pairs (${rawPairs.size}) — pose alignment failed")

    val nP = rawPairs.size
    val srcMat = DenseMatrix.zeros[Double](nP, 3)
    val tgtMat = DenseMatrix.zeros[Double](nP, 3)
    for ((rp, tp), idx) <- rawPairs.zipWithIndex do
      srcMat(idx, 0) = rp.x; srcMat(idx, 1) = rp.y; srcMat(idx, 2) = rp.z
      tgtMat(idx, 0) = tp.x; tgtMat(idx, 1) = tp.y; tgtMat(idx, 2) = tp.z
    val (rotS, scaleS, transS) = WholeCageAlignment.procrustesSimilarity(srcMat, tgtMat)
    val S = Similarity3D(
      rotation = rotS,
      scale = scaleS,
      rotCenter = Point3D(0, 0, 0),
      translation = EuclideanVector3D(transS(0), transS(1), transS(2)),
    )

    val targetInRef = targetCage.transform(p => S.inverse.apply(p))

    val initialOuter = math.max(Geom.boundingBoxDiagonal(reference) / 4.0, 30.0)
    val fittedRef = NonRigid.coarseToFineNonRigidICP(reference, targetInRef, initialOuter, bidirectional)

    val fittedInCage = fittedRef.transform(p => S.apply(p))
    RegisterResult(fitted = fittedInCage, perRib = srcAfterRigid, gpFit = fittedInCage)


// ────────────────────────────────────────────────────────────────────────────
// CLI parsing
// ────────────────────────────────────────────────────────────────────────────

final case class CliArgs(
    inputDir: File,
    outputDir: File,
    templateDir: File,
    templatePid: String,
    nWorkers: Int,
    ribLabels: Seq[Int],
    sides: Seq[String],
    patientFilter: Option[Set[String]],
    skipExisting: Boolean,
    verbose: Boolean,
    bidirectional: Boolean,
    /** When set, additionally dump three per-stage STL sets for this one
     *  patient — used to build the methodology figure. No effect on the
     *  production per-rib output. */
    methodologyPatientId: Option[String],
)

object Cli:
  private val flagArgs = Set("--no-skip-existing", "--verbose", "--bidirectional")

  def parse(args: Seq[String]): CliArgs =
    val argMap = args.toArray.filterNot(flagArgs.contains).grouped(2)
      .collect { case Array(k, v) => k -> v }.toMap

    def required(k: String): String =
      argMap.getOrElse(k, sys.error(s"missing required arg $k"))

    val inputDir = new File(required("--input"))
    val outputDir = new File(required("--output"))
    val templatePid = required("--template-pid")
    val templateDir = new File(argMap.getOrElse("--template-dir", inputDir.getPath))
    val nWorkers = argMap.getOrElse("--workers", "4").toInt
    val ribLabels = argMap.getOrElse(
      "--rib-labels",
      "40,41,42,43,44,45,46,47,48,49,50,51",
    ).split(',').map(_.trim.toInt).toSeq
    val sides = Seq("L", "R")

    val patientFilter: Option[Set[String]] =
      argMap.get("--patient-ids-file").map { p =>
        scala.io.Source.fromFile(p).getLines()
          .map(_.trim).filter(_.nonEmpty).toSet
      }.orElse {
        argMap.get("--patient-ids").map(_.split(',').map(_.trim).toSet)
      }

    if !inputDir.isDirectory then sys.error(s"--input not a directory: $inputDir")
    if !templateDir.isDirectory then sys.error(s"--template-dir not a directory: $templateDir")
    outputDir.mkdirs()

    CliArgs(
      inputDir = inputDir,
      outputDir = outputDir,
      templateDir = templateDir,
      templatePid = templatePid,
      nWorkers = nWorkers,
      ribLabels = ribLabels,
      sides = sides,
      patientFilter = patientFilter,
      skipExisting = !args.contains("--no-skip-existing"),
      verbose = args.contains("--verbose"),
      bidirectional = args.contains("--bidirectional"),
      methodologyPatientId = argMap.get("--methodology-patient-id").map(_.trim).filter(_.nonEmpty),
    )


// ────────────────────────────────────────────────────────────────────────────
// Driver
// ────────────────────────────────────────────────────────────────────────────

object RibRegistration:

  /** Sharded scan of ``<root>/<block>/<pid>/`` for patient IDs. */
  def discoverPatients(root: File): Seq[String] =
    val out = scala.collection.mutable.Buffer.empty[String]
    val blocks = Option(root.listFiles).getOrElse(Array.empty[File])
      .filter(f => f.isDirectory && f.getName.forall(_.isDigit))
    for blockDir <- blocks do
      val pids = Option(blockDir.listFiles).getOrElse(Array.empty[File])
        .filter(f => f.isDirectory && f.getName.forall(_.isDigit))
      out ++= pids.map(_.getName)
    out.distinct.sorted.toSeq

  def loadTemplateMeshes(
      templateDir: File,
      templatePid: String,
      ribLabels: Seq[Int],
      sides: Seq[String],
  ): Map[String, TriangleMesh[_3D]] =
    val pdir = PatientPath.patientDir(templateDir, templatePid)
    val out = scala.collection.mutable.Map.empty[String, TriangleMesh[_3D]]
    for label <- ribLabels; side <- sides do
      val ribId = s"rib${label}_$side"
      val f = new File(pdir, s"${templatePid}_${ribId}.stl")
      if f.exists() then
        MeshIO.readMesh(f) match
          case Success(m) => out(ribId) = m
          case Failure(e) =>
            Log.warn(s"template $ribId unreadable: ${e.getMessage}")
      else
        Log.warn(s"template $ribId missing: $f")
    out.toMap

  def main(args: Array[String]): Unit =
    val cli = Cli.parse(args.toIndexedSeq)
    Log.info(s"input=${cli.inputDir} output=${cli.outputDir}")
    Log.info(s"template-pid=${cli.templatePid} template-dir=${cli.templateDir}")
    Log.info(s"workers=${cli.nWorkers} ribs=${cli.ribLabels.mkString(",")} sides=${cli.sides.mkString(",")} bidirectional=${cli.bidirectional}")
    cli.methodologyPatientId.foreach(p => Log.info(s"methodology-patient-id=$p (will dump _methodology/{cage_patient,per_rib_template,gp_fit_template}/)"))

    val allPids = discoverPatients(cli.inputDir).filter(_ != cli.templatePid)
    val patientPids = cli.patientFilter
      .map(filter => allPids.filter(filter.contains))
      .getOrElse(allPids)
    Log.info(s"discovered ${allPids.size} patients; processing ${patientPids.size}")
    val templates = loadTemplateMeshes(cli.templateDir, cli.templatePid, cli.ribLabels, cli.sides)
    Log.info(s"loaded ${templates.size} template ribs")

    val orderedRibIds: Seq[String] =
      cli.ribLabels.flatMap(l => cli.sides.map(s => s"rib${l}_$s"))

    // ── STAGE 1: whole-cage Procrustes (parallel over patients) ─────────
    Log.info(s"computing whole-cage transforms for ${patientPids.size} patients")
    val wcStartMs = System.currentTimeMillis()
    val templateCentroids = WholeCageAlignment.computeRibCentroids(
      cli.templateDir, cli.templatePid, cli.ribLabels, cli.sides)
    val wcBar = new ProgressBar("whole-cage", patientPids.size)

    val wcExecutor = Executors.newFixedThreadPool(cli.nWorkers)
    given wcEc: ExecutionContext = ExecutionContext.fromExecutorService(wcExecutor)

    val wcFutures: Seq[Future[(String, Option[(DenseMatrix[Double], DenseVector[Double])])]] =
      patientPids.map { pid =>
        Future {
          val patCentroids = WholeCageAlignment.computeRibCentroids(
            cli.inputDir, pid, cli.ribLabels, cli.sides)
          val res = WholeCageAlignment.alignToTemplate(
            templateCentroids, patCentroids, orderedRibIds)
          wcBar.tick()
          (pid, res)
        }
      }

    val wcResults =
      try Await.result(Future.sequence(wcFutures), Duration.Inf)
      finally
        wcExecutor.shutdown()
        wcExecutor.awaitTermination(10, TimeUnit.SECONDS)
    wcBar.finish()

    val wcTransforms: Map[String, (DenseMatrix[Double], DenseVector[Double])] =
      wcResults.collect { case (pid, Some(rt)) => pid -> rt }.toMap
    val wcSkipped = wcResults.count(_._2.isEmpty)
    Log.info(s"whole-cage: aligned=${wcTransforms.size} skipped=$wcSkipped " +
             s"elapsed=${FmtTime(System.currentTimeMillis() - wcStartMs)}")

    // ── STAGE 2: per-rib outer × per-patient parallel inner ─────────────
    val totalStartMs = System.currentTimeMillis()
    var totalDone = 0
    var totalFailed = 0
    for label <- cli.ribLabels; side <- cli.sides do
      val ribId = s"rib${label}_$side"
      templates.get(ribId) match
        case None =>
          Log.warn(s"--- $ribId --- SKIP (template missing)")
        case Some(reference) =>
          val refDiag = Geom.boundingBoxDiagonal(reference)
          Log.info(f"--- $ribId (template: ${reference.pointSet.numberOfPoints} pts, " +
                   f"bbDiag=${refDiag}%.1f mm) ---")

          val todo = patientPids.flatMap { pid =>
            val stl = new File(PatientPath.patientDir(cli.inputDir, pid), s"${pid}_${ribId}.stl")
            if !stl.exists() then None
            else
              val outFile = new File(PatientPath.patientDir(cli.outputDir, pid), s"${pid}_${ribId}.stl")
              if cli.skipExisting && outFile.exists() then None
              else Some((pid, stl, outFile))
          }

          if todo.isEmpty then
            Log.info(s"$ribId: nothing to do (all outputs exist or no input STLs)")
          else
            val ribStartMs = System.currentTimeMillis()
            val ribFailed  = new AtomicInteger(0)
            val total      = todo.size
            val bar        = new ProgressBar(ribId, total)

            val executor = Executors.newFixedThreadPool(cli.nWorkers)
            given ec: ExecutionContext = ExecutionContext.fromExecutorService(executor)

            val futures: Seq[Future[Unit]] = todo.map { case (pid, stlFile, outFile) =>
              Future {
                Try {
                  val targetRaw = MeshIO.readMesh(stlFile).get
                  val targetCage = wcTransforms.get(pid).map { case (r, t) =>
                    WholeCageAlignment.transformMesh(targetRaw, r, t)
                  }.getOrElse(targetRaw)

                  given Random = Random(42)
                  val regResult = Register.registerOne(reference, targetCage, cli.bidirectional)
                  val fittedInCage = regResult.fitted

                  val fittedInPatient = wcTransforms.get(pid).map { case (r, t) =>
                    val rInv = r.t
                    val tInv = -rInv * t
                    WholeCageAlignment.transformMesh(fittedInCage, rInv, tInv)
                  }.getOrElse(fittedInCage)

                  outFile.getParentFile.mkdirs()
                  MeshIO.writeMesh(fittedInPatient, outFile).get

                  // Methodology-figure dumps: only for the designated display
                  // patient, so production runs don't pay any disk cost.
                  if cli.methodologyPatientId.contains(pid) then
                    val methodRoot = new File(outFile.getParentFile, "_methodology")
                    val name       = s"${pid}_${ribId}.stl"
                    val cageDir    = new File(methodRoot, "cage_patient")
                    val perRibDir  = new File(methodRoot, "per_rib_template")
                    val gpDir      = new File(methodRoot, "gp_fit_template")
                    cageDir.mkdirs(); perRibDir.mkdirs(); gpDir.mkdirs()
                    MeshIO.writeMesh(targetCage,      new File(cageDir,   name)).get
                    MeshIO.writeMesh(regResult.perRib, new File(perRibDir, name)).get
                    MeshIO.writeMesh(regResult.gpFit,  new File(gpDir,     name)).get
                } match
                  case Success(_) => ()
                  case Failure(e) =>
                    ribFailed.incrementAndGet()
                    bar.log("ERROR", s"$ribId pid=$pid : ${e.getMessage}")

                bar.tick()
              }
            }

            try Await.result(Future.sequence(futures), Duration.Inf)
            finally
              executor.shutdown()
              executor.awaitTermination(10, TimeUnit.SECONDS)
            bar.finish()

            val ribElapsed = System.currentTimeMillis() - ribStartMs
            val failed     = ribFailed.get()
            val succeeded  = total - failed
            Log.info(s"$ribId: success=$succeeded failed=$failed elapsed=${FmtTime(ribElapsed)}")

            totalDone   += succeeded
            totalFailed += failed

    val totalElapsed = System.currentTimeMillis() - totalStartMs
    Log.info(s"DONE total=$totalDone failed=$totalFailed elapsed=${FmtTime(totalElapsed)}")

    // Scalismo can leave non-daemon threads running; force-exit the JVM.
    sys.exit(0)
