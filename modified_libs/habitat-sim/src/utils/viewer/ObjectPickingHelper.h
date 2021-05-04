#pragma once
#include <Magnum/GL/Framebuffer.h>
#include <Magnum/GL/Renderbuffer.h>
#include <Magnum/Magnum.h>
#include <memory>
#include "esp/gfx/Drawable.h"
#include "esp/gfx/MeshVisualizerDrawable.h"

class ObjectPickingHelper {
 public:
  /**
   * @brief Constructor
   * @param viewportSize the size of the viewport in w x h format
   */
  ObjectPickingHelper(Magnum::Vector2i viewportSize);
  ~ObjectPickingHelper(){};

  /**
   * @brief  Recreate object ID reading renderbuffers that depend on viewport
   * size
   * @param viewportSize the size of the viewport in w x h format
   */
  void recreateFramebuffer(Magnum::Vector2i viewportSize);

  /**
   *@brief Prepare to draw: it will bind the framebuffer, map color attachement,
   *clear depth, background color
   */
  ObjectPickingHelper& prepareToDraw();

  /**
   * @brief set viewport for framebuffer, which must be called in viewport event
   */
  ObjectPickingHelper& handleViewportChange(Magnum::Vector2i viewportSize);
  /**
   * @brief get the object id, aka, the drawable id defined in Drawable class
   * @param eventPosition, mouse event position
   * @param windowSize, the size of the GUI window
   */
  unsigned int getObjectId(const Magnum::Vector2i& mouseEventPosition,
                           const Magnum::Vector2i& windowSize);

  /**
   * @brief create a mesh visualizer (a drawable) for the picked object (a
   * drawable), if there is any
   * @param pickedObject, the picked object (drawable), if nullptr is passed,
   * function will delete any previously created visualizer object
   */
  void createPickedObjectVisualizer(esp::gfx::Drawable* pickedObject);

  /**
   * @brief return true if an object is picked at the moment
   */
  bool isObjectPicked() const { return meshVisualizerDrawable_ != nullptr; }

  esp::gfx::DrawableGroup& getDrawables() { return pickedObjectDrawbles_; }

 protected:
  // framebuffer for drawable selection
  Magnum::GL::Framebuffer selectionFramebuffer_{Magnum::NoCreate};
  Magnum::GL::Renderbuffer selectionDepth_;
  Magnum::GL::Renderbuffer selectionDrawableId_;

  Magnum::Shaders::MeshVisualizer3D shader_{
      Magnum::Shaders::MeshVisualizer3D::Flag::Wireframe};
  esp::gfx::MeshVisualizerDrawable* meshVisualizerDrawable_ = nullptr;
  esp::gfx::DrawableGroup pickedObjectDrawbles_;
  ObjectPickingHelper& mapForDraw();
};
