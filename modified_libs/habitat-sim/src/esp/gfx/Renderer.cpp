// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "Renderer.h"

#include <Corrade/Containers/StridedArrayView.h>
#include <Magnum/GL/Buffer.h>
#include <Magnum/GL/DefaultFramebuffer.h>
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/PixelFormat.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/GL/RenderbufferFormat.h>
#include <Magnum/GL/Renderer.h>
#include <Magnum/GL/Texture.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/Image.h>
#include <Magnum/PixelFormat.h>

#include "esp/gfx/DepthUnprojection.h"
#include "esp/gfx/magnum.h"

namespace Mn = Magnum;

namespace esp {
namespace gfx {

struct Renderer::Impl {
  Impl() {
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::DepthTest);
    Mn::GL::Renderer::enable(Mn::GL::Renderer::Feature::FaceCulling);
  }
  ~Impl() { LOG(INFO) << "Deconstructing Renderer"; }

  void draw(RenderCamera& camera,
            scene::SceneGraph& sceneGraph,
            RenderCamera::Flags flags) {
    for (auto& it : sceneGraph.getDrawableGroups()) {
      // TODO: remove || true
      if (it.second.prepareForDraw(camera) || true) {
        camera.draw(it.second, flags);
      }
    }
  }

  void draw(sensor::VisualSensor& visualSensor,
            scene::SceneGraph& sceneGraph,
            RenderCamera::Flags flags) {
    ASSERT(visualSensor.isVisualSensor());

    // set the modelview matrix, projection matrix of the render camera;
    sceneGraph.setDefaultRenderCamera(visualSensor);

    draw(sceneGraph.getDefaultRenderCamera(), sceneGraph, flags);
  }

  void bindRenderTarget(sensor::VisualSensor& sensor) {
    auto depthUnprojection = sensor.depthUnprojection();
    if (!depthUnprojection) {
      throw std::runtime_error(
          "Sensor does not have a depthUnprojection matrix");
    }

    if (!depthShader_) {
      depthShader_ = std::make_unique<DepthShader>(
          DepthShader::Flag::UnprojectExistingDepth);
    }

    sensor.bindRenderTarget(RenderTarget::create_unique(
        sensor.framebufferSize(), *depthUnprojection, depthShader_.get()));
  }

 private:
  std::unique_ptr<DepthShader> depthShader_ = nullptr;
};

Renderer::Renderer() : pimpl_(spimpl::make_unique_impl<Impl>()) {}

void Renderer::draw(RenderCamera& camera,
                    scene::SceneGraph& sceneGraph,
                    RenderCamera::Flags flags) {
  pimpl_->draw(camera, sceneGraph, flags);
}

void Renderer::draw(sensor::VisualSensor& visualSensor,
                    scene::SceneGraph& sceneGraph,
                    RenderCamera::Flags flags) {
  pimpl_->draw(visualSensor, sceneGraph, flags);
}

void Renderer::bindRenderTarget(sensor::VisualSensor& sensor) {
  pimpl_->bindRenderTarget(sensor);
}

}  // namespace gfx
}  // namespace esp
