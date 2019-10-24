#include "tet_mesh.h"

namespace polyscope {

const std::string TetMesh::structureTypeName = "Tet Mesh";
std::string TetMesh::typeName() { return structureTypeName; }

TetMesh::TetMesh(std::string name_)
    : QuantityStructure<TetMesh>(name_), name(name_) {}

TetMesh::TetMesh(std::string name_, std::vector<glm::vec3> vertices_,
                 std::vector<std::vector<size_t>> tets_)
    : QuantityStructure<TetMesh>(name_), vertices(vertices_), tets(tets_),
      name(name_) {

    for (size_t i = 0; i < tets.size(); ++i) {
        std::vector<size_t> t = tets[i];
        localFaces.emplace_back(std::vector<size_t>{0, 1, 2});
        localFaces.emplace_back(std::vector<size_t>{0, 2, 3});
        localFaces.emplace_back(std::vector<size_t>{0, 3, 1});
        localFaces.emplace_back(std::vector<size_t>{2, 1, 3});

        faces.emplace_back(std::vector<size_t>{t[0], t[1], t[2]});
        faces.emplace_back(std::vector<size_t>{t[0], t[2], t[3]});
        faces.emplace_back(std::vector<size_t>{t[0], t[3], t[1]});
        faces.emplace_back(std::vector<size_t>{t[2], t[1], t[3]});
    }
    computeGeometryData();

    baseColor = getNextUniqueColor();
    tetColor  = baseColor;
    edgeColor = glm::vec3{0, 0, 0};
}

void TetMesh::computeGeometryData() {
    const glm::vec3 zero{0., 0., 0.};

    // Reset face-valued
    faceAreas.resize(faces.size());
    faceNormals.resize(faces.size());

    // Loop over faces to compute face-valued quantities
    for (size_t iF = 0; iF < faces.size(); iF++) {
        auto& face = faces[iF];
        assert(face.size() == 3);

        glm::vec3 fN = zero;
        double fA    = 0;
        glm::vec3 pA = vertices[face[0]];
        glm::vec3 pB = vertices[face[1]];
        glm::vec3 pC = vertices[face[2]];

        fN = glm::cross(pB - pA, pC - pA);
        fA = 0.5 * glm::length(fN);

        // Set face values
        fN              = glm::normalize(fN);
        faceNormals[iF] = fN;
        faceAreas[iF]   = fA;
    }

    // Loop over tets to compute centers and vertex areas
    vertexVolumes = std::vector<double>(vertices.size(), 0.0);
    for (size_t iT = 0; iT < tets.size(); ++iT) {
        glm::vec3 p0     = vertices[tets[iT][0]];
        glm::vec3 p1     = vertices[tets[iT][1]];
        glm::vec3 p2     = vertices[tets[iT][2]];
        glm::vec3 p3     = vertices[tets[iT][3]];
        glm::vec3 center = (p0 + p1 + p2 + p3) / 4.f;
        tetCenters.emplace_back(center);

        double vol = glm::dot(p0 - p3, glm::cross(p1 - p3, p2 - p3)) / 6;
        for (size_t i = 0; i < 4; ++i) {
            vertexVolumes[tets[iT][i]] += vol / 4;
        }
    }
}


glm::vec3 TetMesh::faceCenter(size_t iF) {
    std::vector<size_t> f = faces[iF];
    glm::vec3 center      = vertices[f[0]] + vertices[f[1]] + vertices[f[2]];
    return center / 3.f;
}

size_t TetMesh::nVertices() { return vertices.size(); }
size_t TetMesh::nFaces() { return faces.size(); }
size_t TetMesh::nTets() { return tets.size(); }

void TetMesh::draw() {
    if (!enabled) {
        return;
    }

    if (dominantQuantity == nullptr) {
        if (program == nullptr) {
            prepare();
        }

        // Set uniforms
        setTransformUniforms(*program);
        program->setUniform("u_basecolor", tetColor);

        program->draw();
    }

    // Draw the quantities
    for (auto& x : quantities) {
        x.second->draw();
    }

    // Draw the wireframe
    if (edgeWidth > 0) {
        if (wireframeProgram == nullptr) {
            prepareWireframe();
        }

        // Set uniforms
        setTransformUniforms(*wireframeProgram);
        wireframeProgram->setUniform("u_edgeWidth", edgeWidth);
        wireframeProgram->setUniform("u_edgeColor", edgeColor);

        glEnable(GL_BLEND);
        glDepthFunc(GL_LEQUAL); // Make sure wireframe wins depth tests
        glBlendFuncSeparate(
            GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA, GL_ZERO,
            GL_ONE); // slightly weird blend function: ensures alpha is set by
                     // whatever was drawn before, rather than the wireframe

        wireframeProgram->draw();

        glDepthFunc(GL_LESS); // return to normal
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }
}

void TetMesh::drawPick() {
    if (!enabled) {
        return;
    }

    if (pickProgram == nullptr) {
        preparePick();
    }

    // Set uniforms
    setTransformUniforms(*pickProgram);

    pickProgram->draw();
}

void TetMesh::prepare() {
    program.reset(new gl::GLProgram(&gl::PLAIN_SURFACE_VERT_SHADER,
                                    &gl::PLAIN_SURFACE_FRAG_SHADER,
                                    gl::DrawMode::Triangles));

    // Populate draw buffers
    fillGeometryBuffers(*program);

    setMaterialForProgram(*program, "wax");
}

void TetMesh::prepareWireframe() {
    wireframeProgram.reset(new gl::GLProgram(&gl::SURFACE_WIREFRAME_VERT_SHADER,
                                             &gl::SURFACE_WIREFRAME_FRAG_SHADER,
                                             gl::DrawMode::Triangles));

    // Populate draw buffers
    fillGeometryBuffersWireframe(*wireframeProgram);

    setMaterialForProgram(*wireframeProgram, "wax");
}

void TetMesh::preparePick() {

    // Create a new program
    pickProgram.reset(new gl::GLProgram(&gl::PICK_SURFACE_VERT_SHADER,
                                        &gl::PICK_SURFACE_FRAG_SHADER,
                                        gl::DrawMode::Triangles));

    fillGeometryBuffersPick(*pickProgram);
}

void TetMesh::sliceTet( const std::array<glm::vec3,4>& p, // in: tet vertices
                        glm::vec3 N,                     // in: plane normal
                        double c,                      // in: plane offset
                        std::vector<glm::vec3>& q,        // out: intersection points
                        std::vector<std::pair<size_t,size_t>>& e, // out: intersection edges
                        std::vector<double>& T )       // out: intersection times
{
    std::array<double,4> d;
    for( size_t i = 0; i < 4; i++ ) {
      d[i] = dot(N,p[i]) - c;
    }
     size_t n = 0;
     for( size_t i = 0; i < 4; i++ ) {
       for( size_t j = i+1; j < 4; j++ ) {
         if( d[i]*d[j] < 0. ) {
           float t = (0-d[i])/(d[j]-d[i]);
           q.push_back( (1.f-t)*p[i] + t*p[j] );
           e.push_back( std::pair<int,int>(i,j) );
           T.push_back( t );
         }
         n++;
       }
     }
     if( q.size() == 4 &&
           dot( cross( q[1]-q[0], q[2]-q[0] ),
                cross( q[2]-q[0], q[3]-q[0] ) ) < 0. )
     {
       std::swap( q[2], q[3] );
       std::swap( e[2], e[3] );
       std::swap( T[2], T[3] );
     }
}

void TetMesh::fillGeometryBuffersPick(gl::GLProgram& p) {
    // Get element indices
    size_t totalPickElements = nVertices() + nFaces();

    // In "local" indices, indexing elements only within this triMesh, used for
    // reading later
    facePickIndStart = nVertices();

    // In "global" indices, indexing all elements in the scene, used to fill
    // buffers for drawing here
    size_t pickStart = pick::requestPickBufferRange(this, totalPickElements);
    size_t faceGlobalPickIndStart = pickStart + nVertices();

    // == Fill buffers
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> bcoord;
    std::vector<std::array<glm::vec3, 3>> vertexColors, edgeColors,
        halfedgeColors;
    std::vector<glm::vec3> faceColor;

    // Reserve space
    positions.reserve(3 * nFaces());
    bcoord.reserve(3 * nFaces());
    vertexColors.reserve(3 * nFaces());
    edgeColors.reserve(3 * nFaces());
    halfedgeColors.reserve(3 * nFaces());
    faceColor.reserve(3 * nFaces());

    // for now, we just color edges and halfedges their respective face's colors
    // TODO: figure out edges

    // Build all quantities in each face
    for (size_t iT = 0; iT < tets.size(); iT++) {
        if (glm::dot(tetCenters[iT], sliceNormal) > sliceDist) continue;
        for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            auto& face = faces[iF];

            size_t vA    = face[0];
            glm::vec3 pA = vertices[vA];
            size_t vB    = face[1];
            glm::vec3 pB = vertices[vB];
            size_t vC    = face[2];
            glm::vec3 pC = vertices[vC];

            glm::vec3 fColor = pick::indToVec(iF + faceGlobalPickIndStart);
            std::array<size_t, 3> vertexInds = {vA, vB, vC};

            positions.push_back(pA);
            positions.push_back(pB);
            positions.push_back(pC);

            // Build all quantities
            std::array<glm::vec3, 3> vColor;
            std::array<glm::vec3, 3> constFColor;

            for (size_t i = 0; i < 3; i++) {
                // Want just one copy of face color, so we can build it in the
                // usual way
                faceColor.push_back(fColor);
                constFColor[i] = fColor;

                // Vertex index color
                vColor[i] = pick::indToVec(vertexInds[i] + pickStart);
            }


            // Push three copies of the values needed at each vertex
            for (int j = 0; j < 3; j++) {
                vertexColors.push_back(vColor);
                edgeColors.push_back(constFColor);
                halfedgeColors.push_back(constFColor);
            }

            // Barycoords
            bcoord.push_back(glm::vec3{1.0, 0.0, 0.0});
            bcoord.push_back(glm::vec3{0.0, 1.0, 0.0});
            bcoord.push_back(glm::vec3{0.0, 0.0, 1.0});
        }
    }

    // Store data in buffers
    pickProgram->setAttribute("a_position", positions);
    pickProgram->setAttribute("a_barycoord", bcoord);
    pickProgram->setAttribute<glm::vec3, 3>("a_vertexColors", vertexColors);
    pickProgram->setAttribute<glm::vec3, 3>("a_edgeColors", edgeColors);
    pickProgram->setAttribute<glm::vec3, 3>("a_halfedgeColors", halfedgeColors);
    pickProgram->setAttribute("a_faceColor", faceColor);
}

void TetMesh::fillGeometryBuffers(gl::GLProgram& p) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> bcoord;

    bool wantsBary = p.hasAttribute("a_barycoord");

    positions.reserve(3 * faces.size());
    normals.reserve(3 * faces.size());
    if (wantsBary) {
        bcoord.reserve(3 * faces.size());
    }

    for (size_t iT = 0; iT < tets.size(); iT++) {

      std::vector<glm::vec3> tetPositions;
      std::vector<std::vector<size_t>> tetFaces;
      std::vector<glm::vec3> tetNormals;

      std::vector<size_t> t = tets[iT];
      if (sliceThroughTets) {
        std::array<glm::vec3, 4> p{vertices[t[0]],
                                   vertices[t[1]],
                                   vertices[t[2]],
                                   vertices[t[3]]};
        std::vector<glm::vec3> q;
        std::vector<std::pair<size_t, size_t>> e;
        std::vector<double> t;
        sliceTet(p, sliceNormal, sliceDist, q, e, t);

        if (q.size() == 0) {
          if (glm::dot(p[0], sliceNormal) <= sliceDist) {
            tetPositions = std::vector<glm::vec3>(p.begin(), p.end());
            for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
              tetFaces.emplace_back(localFaces[iF]);
              tetNormals.emplace_back(faceNormals[iF]);
            }
          }
        } else if (q.size() == 3) {
          
        } else if (q.size() == 4) {
        } else {
          std::cerr << "It should be impossible to have a plane intersect " << q.size()
                    << " edges, but it happened." << std::endl;
          exit(1);
        }

      } else {

        if (glm::dot(tetCenters[iT], sliceNormal) <= sliceDist) {
          tetPositions = std::vector<glm::vec3>{vertices[t[0]],
                                                vertices[t[1]],
                                                vertices[t[2]],
                                                vertices[t[3]]};
          for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            tetFaces.emplace_back(localFaces[iF]);
            tetNormals.emplace_back(faceNormals[iF]);
          }
        }
      }

      for (size_t iF = 0; iF < tetFaces.size(); ++iF) {
          auto& face = tetFaces[iF];
          size_t D = face.size();
          glm::vec3 faceN = tetNormals[iF];

          // implicitly triangulate from root
          glm::vec3 pRoot = tetPositions[face[0]];
          for (size_t j = 1; (j + 1) < D; j++) {
            glm::vec3 pB = tetPositions[face[j]];
            glm::vec3 pC = tetPositions[face[j+1]];

            positions.push_back(pRoot);
            positions.push_back(pB);
            positions.push_back(pC);

            normals.push_back(faceN);
            normals.push_back(faceN);
            normals.push_back(faceN);

            if (wantsBary) {
              bcoord.push_back(glm::vec3{1., 0., 0.});
              bcoord.push_back(glm::vec3{0., 1., 0.});
              bcoord.push_back(glm::vec3{0., 0., 1.});
            }
          }
      }
    }

    // Store data in buffers
    p.setAttribute("a_position", positions);
    p.setAttribute("a_normal", normals);
    if (wantsBary) {
        p.setAttribute("a_barycoord", bcoord);
    }
}

void TetMesh::fillGeometryBuffersWireframe(gl::GLProgram& p) {
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec3> bcoord;
    std::vector<glm::vec3> edgeReal;

    positions.reserve(3 * faces.size());
    normals.reserve(3 * faces.size());
    bcoord.reserve(3 * faces.size());
    edgeReal.reserve(3 * faces.size());

    for (size_t iT = 0; iT < tets.size(); iT++) {
        if (glm::dot(tetCenters[iT], sliceNormal) > sliceDist) continue;
        for (size_t iF = 4 * iT; iF < 4 * iT + 4; ++iF) {
            auto& face      = faces[iF];
            size_t D        = face.size();
            glm::vec3 faceN = faceNormals[iF];

            // implicitly triangulate from root
            size_t vA    = face[0];
            glm::vec3 pA = vertices[vA];
            size_t vB    = face[1];
            glm::vec3 pB = vertices[vB];
            size_t vC    = face[2];
            glm::vec3 pC = vertices[vC];

            positions.push_back(pA);
            positions.push_back(pB);
            positions.push_back(pC);

            normals.push_back(faceN);
            normals.push_back(faceN);
            normals.push_back(faceN);

            bcoord.push_back(glm::vec3{1., 0., 0.});
            bcoord.push_back(glm::vec3{0., 1., 0.});
            bcoord.push_back(glm::vec3{0., 0., 1.});

            glm::vec3 edgeRealV{1., 1., 1.};
            edgeReal.push_back(edgeRealV);
            edgeReal.push_back(edgeRealV);
            edgeReal.push_back(edgeRealV);
        }
    }


    // Store data in buffers
    p.setAttribute("a_position", positions);
    p.setAttribute("a_normal", normals);
    p.setAttribute("a_barycoord", bcoord);
    p.setAttribute("a_edgeReal", edgeReal);
}

// == Build the ImGUI ui elements
void TetMesh::buildCustomUI() {
    // Print stats
    long long int nVertsL = static_cast<long long int>(vertices.size());
    long long int nFacesL = static_cast<long long int>(faces.size());
    long long int nTetsL  = static_cast<long long int>(tets.size());
    ImGui::Text("#verts: %lld  #faces: %lld\n#tets: %lld", nVertsL, nFacesL,
                nTetsL);

    { // colors
        ImGui::ColorEdit3("Color", (float*)&tetColor,
                          ImGuiColorEditFlags_NoInputs);
        ImGui::SameLine();
        ImGui::PushItemWidth(100);
        ImGui::ColorEdit3("Edge Color", (float*)&edgeColor,
                          ImGuiColorEditFlags_NoInputs);
        ImGui::PopItemWidth();
    }

    { // Edge width
        ImGui::PushItemWidth(100);
        ImGui::SliderFloat("Edge Width", &edgeWidth, 0.0, 1., "%.5f", 2.);
        if (ImGui::SliderFloat("Plane Distance", &sliceDist, -1., 1., "%.5f",
                               2.)) {
            geometryChanged();
        }
        if (ImGui::SliderFloat("Plane theta", &sliceTheta, 0., 2. * PI, "%.5f",
                               2.)) {
            sliceNormal =
                glm::vec3{cos(sliceTheta) * cos(slicePhi), sin(slicePhi),
                          sin(sliceTheta) * cos(slicePhi)};
            geometryChanged();
        }
        if (ImGui::SliderFloat("Plane phi", &slicePhi, 0., 2. * PI, "%.5f",
                               2.)) {
            sliceNormal =
                glm::vec3{cos(sliceTheta) * cos(slicePhi), sin(slicePhi),
                          sin(sliceTheta) * cos(slicePhi)};
            geometryChanged();
        }
        if (ImGui::Checkbox("Slice through tets", &sliceThroughTets)) {
            geometryChanged();
        }
        ImGui::PopItemWidth();
    }
}
void TetMesh::buildCustomOptionsUI(){};

// Draw pick UI elements when index localPickID is
// selected
// TODO: add GUI for quantities
void TetMesh::buildPickUI(size_t localPickID) {
    // Selection type
    if (localPickID < facePickIndStart) {
        buildVertexInfoGui(localPickID);
    } else {
        buildFaceInfoGui(localPickID - facePickIndStart);
    }
};

void TetMesh::buildVertexInfoGui(size_t vInd) {

    size_t displayInd = vInd;
    ImGui::TextUnformatted(("Vertex #" + std::to_string(displayInd)).c_str());

    std::stringstream buffer;
    buffer << vertices[vInd];
    ImGui::TextUnformatted(("Position: " + buffer.str()).c_str());

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Indent(20.);

    // Build GUI to show the quantities
    ImGui::Columns(2);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / 3);
    for (auto& x : quantities) {
        x.second->buildVertexInfoGUI(vInd);
    }

    ImGui::Indent(-20.);
}

void TetMesh::buildFaceInfoGui(size_t fInd) {
    size_t displayInd = fInd;
    ImGui::TextUnformatted(("Face #" + std::to_string(displayInd)).c_str());

    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Spacing();
    ImGui::Indent(20.);

    // Build GUI to show the quantities
    ImGui::Columns(2);
    ImGui::SetColumnWidth(0, ImGui::GetWindowWidth() / 3);
    for (auto& x : quantities) {
        x.second->buildFaceInfoGUI(fInd);
    }

    ImGui::Indent(-20.);
}


// = Length and bounding box (returned in object coordinates)
std::tuple<glm::vec3, glm::vec3> TetMesh::boundingBox() {
    glm::vec3 min{100, 100, 100};
    glm::vec3 max{-100, -100, -100};
    for (glm::vec3 v : vertices) {
        min.x = fmin(min.x, v.x);
        min.y = fmin(min.y, v.y);
        min.z = fmin(min.z, v.z);
        max.x = fmax(max.x, v.x);
        max.y = fmax(max.y, v.y);
        max.z = fmax(max.z, v.z);
    }
    return std::make_tuple(min, max);
}
double TetMesh::lengthScale() {
    std::tuple<glm::vec3, glm::vec3> bb = boundingBox();
    return glm::distance(std::get<1>(bb), std::get<0>(bb));
}


void TetMesh::geometryChanged() {
    program.reset();
    wireframeProgram.reset();
    pickProgram.reset();

    computeGeometryData();

    for (auto& q : quantities) {
        q.second->geometryChanged();
    }
}

TetVertexScalarQuantity* TetMesh::addVertexScalarQuantityImpl(
    std::string name, const std::vector<double>& data, DataType type) {
    TetVertexScalarQuantity* q =
        new TetVertexScalarQuantity(name, data, *this, type);
    addQuantity(q);
    return q;
}

TetFaceScalarQuantity* TetMesh::addFaceScalarQuantityImpl(
    std::string name, const std::vector<double>& data, DataType type) {
    TetFaceScalarQuantity* q =
        new TetFaceScalarQuantity(name, data, *this, type);
    addQuantity(q);
    return q;
}


TetVertexVectorQuantity*
TetMesh::addVertexVectorQuantityImpl(std::string name,
                                     const std::vector<glm::vec3>& vectors,
                                     VectorType vectorType) {
    TetVertexVectorQuantity* q =
        new TetVertexVectorQuantity(name, vectors, *this, vectorType);
    addQuantity(q);
    return q;
}

TetFaceVectorQuantity*
TetMesh::addFaceVectorQuantityImpl(std::string name,
                                   const std::vector<glm::vec3>& vectors,
                                   VectorType vectorType) {

    TetFaceVectorQuantity* q =
        new TetFaceVectorQuantity(name, vectors, *this, vectorType);
    addQuantity(q);
    return q;
}

TetTetVectorQuantity*
TetMesh::addTetVectorQuantityImpl(std::string name,
                                  const std::vector<glm::vec3>& vectors,
                                  VectorType vectorType) {

    TetTetVectorQuantity* q =
        new TetTetVectorQuantity(name, vectors, *this, vectorType);
    addQuantity(q);
    return q;
}

TetMeshQuantity::TetMeshQuantity(std::string name, TetMesh& parentStructure,
                                 bool dominates)
    : Quantity<TetMesh>(name, parentStructure, dominates) {}
void TetMeshQuantity::geometryChanged(){};
void TetMeshQuantity::buildVertexInfoGUI(size_t vInd) {}
void TetMeshQuantity::buildFaceInfoGUI(size_t fInd) {}
void TetMeshQuantity::buildEdgeInfoGUI(size_t eInd) {}
void TetMeshQuantity::buildHalfedgeInfoGUI(size_t heInd) {}
void TetMeshQuantity::buildTetInfoGUI(size_t heInd) {}

} // namespace polyscope
