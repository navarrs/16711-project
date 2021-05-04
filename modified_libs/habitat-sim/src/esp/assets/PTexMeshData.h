// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/BufferTexture.h>
#include <Magnum/GL/Mesh.h>
#include <Magnum/GL/Texture.h>

#include "BaseMesh.h"
#include "esp/core/esp.h"

namespace esp {
namespace assets {

class PTexMeshData : public BaseMesh {
 public:
  struct MeshData {
    std::vector<vec3f> vbo;
    std::vector<vec4f> nbo;
    std::vector<vec4uc> cbo;
    std::vector<uint32_t> ibo;
    std::vector<uint32_t> ibo_tri;
  };

  struct RenderingBuffer {
    Magnum::GL::Mesh mesh;
    Magnum::GL::Mesh triangleMesh;
    Magnum::GL::Texture2D atlasTexture;
    Magnum::GL::Buffer vertexBuffer;
    Magnum::GL::Buffer indexBuffer;
    Magnum::GL::Buffer triangleMeshIndexBuffer;
    Magnum::GL::Buffer adjFacesBuffer;
    Magnum::GL::BufferTexture adjFacesBufferTexture;

    RenderingBuffer()
        : adjFacesBuffer{Magnum::GL::Buffer::TargetHint::Texture} {}
  };

  PTexMeshData() : BaseMesh(SupportedMeshType::PTEX_MESH) {}
  virtual ~PTexMeshData(){};

  // ==== geometry ====
  void load(const std::string& meshFile, const std::string& atlasFolder);
  uint32_t tileSize() const { return tileSize_; }

  const std::vector<MeshData>& meshes() const;
  std::string atlasFolder() const;
  void resize(size_t n) { submeshes_.resize(n); }

  int getSize() { return submeshes_.size(); }

  static void parsePLY(const std::string& filename, MeshData& meshData);
  static void calculateAdjacency(const MeshData& mesh,
                                 std::vector<uint32_t>& adjFaces);

  // ==== rendering ====
  RenderingBuffer* getRenderingBuffer(int submeshID);
  virtual void uploadBuffersToGPU(bool forceReload = false) override;
  virtual Magnum::GL::Mesh* getMagnumGLMesh(int submeshID) override;

  float exposure() const;
  void setExposure(float val);

  float gamma() const;
  void setGamma(float val);

  float saturation() const;
  void setSaturation(float val);

 protected:
  void loadMeshData(const std::string& meshFile);

  float splitSize_ = 0.0f;
  uint32_t tileSize_ = 0;

  // initial values are based on ReplicaSDK
  //! @brief exposure, the amount of light per unit area reaching the image
  float exposure_ = 0.0125f;

  //! @brief gamma, the exponent applied in the gamma correction
  float gamma_ = 1.0f / 1.6969f;

  //! @brief saturation, the intensity of a color
  float saturation_ = 1.5f;

  std::string atlasFolder_;
  std::vector<MeshData> submeshes_;

  // ==== rendering ====
  // we will have to use smart pointer here since each item within the structure
  // (e.g., Magnum::GL::Mesh) does NOT have copy constructor
  std::vector<std::unique_ptr<RenderingBuffer>> renderingBuffers_;
};

}  // namespace assets
}  // namespace esp
