// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#ifndef ESP_ASSETS_RESOURCEMANAGER_H_
#define ESP_ASSETS_RESOURCEMANAGER_H_

/** @file
 * @brief Class @ref esp::assets::ResourceManager, enum @ref
 * esp::assets::ResourceManager::ShaderType
 */

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Corrade/Containers/Optional.h>
#include <Magnum/EigenIntegration/Integration.h>
#include <Magnum/GL/TextureFormat.h>
#include <Magnum/MeshTools/Compile.h>
#include <Magnum/MeshTools/Transform.h>
#include <Magnum/SceneGraph/MatrixTransformation3D.h>

#include "Asset.h"
#include "BaseMesh.h"
#include "CollisionMeshData.h"
#include "GenericMeshData.h"
#include "MeshData.h"
#include "MeshMetaData.h"
#include "attributes/AttributesBase.h"
#include "esp/gfx/DrawableGroup.h"
#include "esp/gfx/MaterialData.h"
#include "esp/gfx/ShaderManager.h"
#include "esp/physics/configure.h"
#include "esp/scene/SceneManager.h"
#include "esp/scene/SceneNode.h"

#include "managers/AssetAttributesManager.h"
#include "managers/ObjectAttributesManager.h"
#include "managers/PhysicsAttributesManager.h"
#include "managers/StageAttributesManager.h"

// forward declarations
namespace Magnum {
namespace Trade {
class AbstractImporter;
class AbstractShaderProgram;
class PhongMaterialData;
}  // namespace Trade
}  // namespace Magnum

namespace Mn = Magnum;

namespace Attrs = esp::assets::attributes;

namespace esp {
namespace gfx {
class Drawable;
}
namespace scene {
struct SceneConfiguration;
}
namespace physics {
class PhysicsManager;
class RigidObject;
}  // namespace physics
namespace nav {
class PathFinder;
}
namespace assets {

/**
 * @brief Singleton class responsible for
 * loading and managing common simulator assets such as meshes, textures, and
 * materials.
 */
class ResourceManager {
 public:
  /** @brief Stores references to a set of drawable elements */
  using DrawableGroup = gfx::DrawableGroup;
  /** @brief Convenience typedef for Importer class */
  using Importer = Mn::Trade::AbstractImporter;

  /**
   * @brief The @ref ShaderManager key for @ref LightInfo which has no lights
   */
  static constexpr char NO_LIGHT_KEY[] = "no_lights";

  /**
   *@brief The @ref ShaderManager key for the default @ref LightInfo
   */
  static constexpr char DEFAULT_LIGHTING_KEY[] = "";

  /**
   *@brief The @ref ShaderManager key for the default @ref MaterialInfo
   */
  static constexpr char DEFAULT_MATERIAL_KEY[] = "";

  /**
   *@brief The @ref ShaderManager key for full ambient white @ref MaterialInfo
   *used for primitive wire-meshes
   */
  static constexpr char WHITE_MATERIAL_KEY[] = "ambient_white";

  /**
   *@brief The @ref ShaderManager key for @ref MaterialInfo with per-vertex
   * object ID
   */
  static constexpr char PER_VERTEX_OBJECT_ID_MATERIAL_KEY[] =
      "per_vertex_object_id";

  /** @brief Constructor */
  explicit ResourceManager();

  /** @brief Destructor */
  ~ResourceManager() {}

  /**
   * @brief This function will build the various @ref Importers and @ref
   * AttributesManagers used by the system.
   */
  void buildImportersAndAttributesManagers();

  /**
   * @brief Build default primitive attribute files and synthesize an object of
   * each type.
   */
  void initDefaultPrimAttributes();

  /**
   * @brief Instantiate, or reinstantiate, PhysicsManager defined by passed
   * attributes
   * @param physicsManager The currently defined @ref physics::PhysicsManager.
   * Will be reseated to the specified physics implementation.
   * @param isEnabled Whether this PhysicsManager is enabled or not.  Takes the
   * place of old checks for nullptr.
   * @param parent The @ref scene::SceneNode of which the scene mesh will be
   * added as a child. Typically near the root of the scene. Expected to be
   * static.
   * @param physicsManagerAttributes A smart pointer to meta data structure
   * storing configured physics simulation parameters.
   */
  void initPhysicsManager(
      std::shared_ptr<physics::PhysicsManager>& physicsManager,
      bool isEnabled,
      scene::SceneNode* parent,
      const Attrs::PhysicsManagerAttributes::ptr& physicsManagerAttributes);

  /**
   * @brief Load a scene mesh and add it to the specified @ref DrawableGroup as
   * a child of the specified @ref scene::SceneNode.
   *
   * If parent and drawables are not specified, the assets are loaded, but no
   * new @ref gfx::Drawable is added for the scene (i.e. it will not be
   * rendered).
   * @param sceneAttributes The @ref StageAttributes that describes the
   * scene
   * @param _physicsManager The currently defined @ref physics::PhysicsManager.
   * @param sceneManagerPtr Pointer to scene manager, to fetch drawables and
   * parent node.
   * @param [out] Current active scene ID is in idx 0, if semantic scene is
   * made, its activeID should be pushed onto vector
   * @param createSemanticMesh If the semantic mesh should be created, based on
   * @ref SimulatorConfiguration
   * @return Whether or not the scene load succeeded.
   */
  bool loadStage(const Attrs::StageAttributes::ptr& sceneAttributes,
                 std::shared_ptr<physics::PhysicsManager> _physicsManager,
                 esp::scene::SceneManager* sceneManagerPtr,
                 std::vector<int>& activeSceneIDs,
                 bool createSemanticMesh);

  /**
   * @brief Construct scene collision mesh group based on name and type of
   * scene.
   * @tparam T type of meshdata desired based on scene type.
   * @param filename The name of the file holding the mesh data
   * @param meshGroup The meshgroup to build
   * @return whether built successfully or not
   */
  template <class T>
  bool buildStageCollisionMeshGroup(const std::string& filename,
                                    std::vector<CollisionMeshData>& meshGroup);

  /**
   * @brief Load/instantiate any required render and collision assets for an
   * object, if they do not already exist in @ref resourceDict_ or @ref
   * collisionMeshGroups_, respectively. Assumes valid render and collisions
   * asset handles have been specified (This is checked/verified in
   * @ref registerObjectTemplate())
   * @param objectTemplateHandle The key for referencing the template in the
   * @ref physicsObjTemplateLibrary_.
   * @return whether process succeeded or not - only currently fails if
   * registration call fails.
   */
  bool instantiateAssetsOnDemand(const std::string& objTemplateHandle);

  //======== Accessor functions ========
  /**
   * @brief Getter for all @ref assets::CollisionMeshData associated with the
   * particular asset.
   *
   * @param collisionAssetHandle The key by which the asset is referenced in
   * @ref collisionMeshGroups_, from the @ref physicsObjTemplateLibrary_.
   * @return A vector reference to @ref assets::CollisionMeshData instances for
   * individual components of the asset.
   */
  const std::vector<assets::CollisionMeshData>& getCollisionMesh(
      const std::string& collisionAssetHandle) const {
    CHECK(collisionMeshGroups_.count(collisionAssetHandle) > 0);
    return collisionMeshGroups_.at(collisionAssetHandle);
  }

  /**
   * @brief Return manager for construction and access to asset attributes.
   */
  const managers::AssetAttributesManager::ptr getAssetAttributesManager()
      const {
    return assetAttributesManager_;
  }
  /**
   * @brief Return manager for construction and access to object attributes.
   */
  const managers::ObjectAttributesManager::ptr getObjectAttributesManager()
      const {
    return objectAttributesManager_;
  }
  /**
   * @brief Return manager for construction and access to physics world
   * attributes.
   */
  const managers::PhysicsAttributesManager::ptr getPhysicsAttributesManager()
      const {
    return physicsAttributesManager_;
  }
  /**
   * @brief Return manager for construction and access to scene attributes.
   */
  const managers::StageAttributesManager::ptr getStageAttributesManager()
      const {
    return stageAttributesManager_;
  }

  /**
   * @brief Retrieve the composition of all transforms applied to a mesh
   * since it was loaded.
   *
   * See @ref translateMesh.
   * @param meshIndex Index of the mesh in @ref meshes_.
   * @return The transformation matrix mapping from the original state to
   * its current state.
   */
  const Mn::Matrix4& getMeshTransformation(const size_t meshIndex) const {
    return meshes_[meshIndex]->meshTransform_;
  }

  /**
   * @brief Retrieve the meta data for a particular asset.
   *
   * This includes identifiers for meshes, textures, materials, and a
   * component heirarchy.
   * @param metaDataName The key identifying the asset in @ref resourceDict_.
   * Typically the filepath of file-based assets.
   * @return The asset's @ref MeshMetaData object.
   */
  const MeshMetaData& getMeshMetaData(const std::string& metaDataName) const {
    CHECK(resourceDict_.count(metaDataName) > 0);
    return resourceDict_.at(metaDataName).meshMetaData;
  }

  /**
   * @brief Get a named @ref LightSetup
   */
  Mn::Resource<gfx::LightSetup> getLightSetup(
      const Mn::ResourceKey& key = Mn::ResourceKey{DEFAULT_LIGHTING_KEY}) {
    return shaderManager_.get<gfx::LightSetup>(key);
  }

  /**
   * @brief Set a named @ref LightSetup
   *
   * If this name already exists, the @ref LightSetup is updated and all @ref
   * Drawables using this setup are updated.
   *
   * @param setup Light setup this key will now reference
   * @param key Key to identify this @ref LightSetup
   */
  void setLightSetup(gfx::LightSetup setup,
                     const Mn::ResourceKey& key = Mn::ResourceKey{
                         DEFAULT_LIGHTING_KEY}) {
    shaderManager_.set(key, std::move(setup), Mn::ResourceDataState::Mutable,
                       Mn::ResourcePolicy::Manual);
  }

  /**
   * @brief Construct a unified @ref MeshData from a loaded asset's collision
   * meshes.
   *
   * See @ref joinHeirarchy.
   * @param filename The identifying string key for the asset. See @ref
   * resourceDict_ and @ref meshes_.
   * @return The unified @ref MeshData object for the asset.
   */
  std::unique_ptr<MeshData> createJoinedCollisionMesh(
      const std::string& filename);

  /**
   * @brief Add an object from a specified object template handle to the
   * specified @ref DrawableGroup as a child of the specified @ref
   * scene::SceneNode if provided.
   *
   * If the attributes specified by objTemplateID exists in @ref
   * physicsObjTemplateLibrary_, and both parent and drawables are
   * specified, than an object referenced by that key is added to the scene.
   * @param objTemplateLibID The ID of the object attributes in the @ref
   * physicsObjTemplateLibrary_.  This is expected to exist
   * @param parent The @ref scene::SceneNode of which the object will be a
   * child.
   * @param drawables The @ref DrawableGroup with which the object @ref
   * gfx::Drawable will be rendered.
   * @param lightSetup The @ref LightSetup key that will be used
   * for the added component.
   * @param[out] visNodeCache Cache for pointers to all nodes created as the
   * result of this process.
   */
  void addObjectToDrawables(int objTemplateLibID,
                            scene::SceneNode* parent,
                            DrawableGroup* drawables,
                            std::vector<scene::SceneNode*>& visNodeCache,
                            const Mn::ResourceKey& lightSetup = Mn::ResourceKey{
                                DEFAULT_LIGHTING_KEY}) {
    if (objTemplateLibID != ID_UNDEFINED) {
      const std::string& objTemplateHandleName =
          objectAttributesManager_->getTemplateHandleByID(objTemplateLibID);

      addObjectToDrawables(objTemplateHandleName, parent, drawables,
                           visNodeCache, lightSetup);
    }  // else objTemplateID does not exist - shouldn't happen
  }    // addObjectToDrawables

  /**
   * @brief Add an object from a specified object template handle to the
   * specified @ref DrawableGroup as a child of the specified @ref
   * scene::SceneNode if provided.
   *
   * If the attributes specified by objTemplateHandle exists in @ref
   * physicsObjTemplateLibrary_, and both parent and drawables are
   * specified, than an object referenced by that key is added to the scene.
   * @param objTemplateHandle The key of the attributes in the @ref  to parse
   * and load.  The attributes are expected to exist but will be created (in the
   * case of synthesized objects) if it does not.
   * @param parent The @ref scene::SceneNode of which the object will be a
   * child.
   * @param drawables The @ref DrawableGroup with which the object @ref
   * gfx::Drawable will be rendered.
   * @param lightSetup The @ref LightSetup key that will be used
   * for the added component.
   * @param[out] visNodeCache Cache for pointers to all nodes created as the
   * result of this process.
   */
  void addObjectToDrawables(const std::string& objTemplateHandle,
                            scene::SceneNode* parent,
                            DrawableGroup* drawables,
                            std::vector<scene::SceneNode*>& visNodeCache,
                            const Mn::ResourceKey& lightSetup = Mn::ResourceKey{
                                DEFAULT_LIGHTING_KEY});

  /**
   * @brief Create a new drawable primitive attached to the desired @ref
   * scene::SceneNode.
   *
   * See @ref primitive_meshes_.
   * @param primitiveID The key of the primitive in @ref primitive_meshes_.
   * @param node The @ref scene::SceneNode to which the primitive drawable
   * will be attached.
   * @param drawables The @ref DrawableGroup with which the primitive will be
   * rendered.
   */
  void addPrimitiveToDrawables(int primitiveID,
                               scene::SceneNode& node,
                               DrawableGroup* drawables);

  /**
   * @brief Remove the specified primitive mesh.
   *
   * @param primitiveID The key of the primitive in @ref primitive_meshes_.
   */
  void removePrimitiveMesh(int primitiveID);

  /**
   * @brief generate a new primitive mesh asset for the NavMesh loaded in the
   * provided PathFinder object.
   *
   * If parent and drawables are provided, create the Drawable and render the
   * NavMesh.
   * @param pathFinder Holds the NavMesh information.
   * @param parent The new Drawable is attached to this node.
   * @param drawables The group with which the new Drawable will be rendered.
   * @return The primitive ID of the new object or @ref ID_UNDEFINED if
   * construction failed.
   */
  int loadNavMeshVisualization(esp::nav::PathFinder& pathFinder,
                               scene::SceneNode* parent,
                               DrawableGroup* drawables);

  /**
   * @brief Build a configuration frame from scene or object attributes values
   * and return it
   *
   * @param attribs the attributes to query for the information.
   * @param origin Either the origin of the sceneAttributes or the COM value of
   * the objectAttributes.
   * @return the coordinate frame of the assets the passed attributes describes.
   */
  esp::geo::CoordinateFrame buildFrameFromAttributes(
      const Attrs::AbstractObjectAttributes::ptr& attribs,
      const Magnum::Vector3& origin);

  /**
   * @brief Set whether textures should be compressed.
   * @param newVal New texture compression setting.
   */
  inline void compressTextures(bool newVal) { compressTextures_ = newVal; };

 private:
  /**
   * @brief Load the requested mesh info into @ref meshInfo corresponding to
   * specified @ref meshType used by @ref objectTemplateHandle
   *
   * @param filename the name of the file describing this mesh
   * @param objectTemplateHandle the handle for the object attributes owning
   * this mesh (for error log output)
   * @param meshType either "render" or "collision" (for error log output)
   * @param requiresLighting whether or not this mesh asset responds to
   * lighting
   * @return whether or not the mesh was loaded successfully
   */
  bool loadObjectMeshDataFromFile(const std::string& filename,
                                  const std::string& objectTemplateHandle,
                                  const std::string& meshType,
                                  const bool requiresLighting);

  /**
   * @brief Build a primitive asset based on passed template parameters.  If
   * exists already, does nothing.  Will use primitiveImporter_ to call
   * appropriate method to construct asset.
   * @param primTemplateHandle the handle referring to the attributes describing
   * primitive to instantiate
   */
  void buildPrimitiveAssetData(const std::string& primTemplateHandle);

 protected:
  // ======== Structs and Types only used locally ========
  /**
   * @brief Data for a loaded asset
   *
   * Contains mesh, texture, material, and asset info
   */
  struct LoadedAssetData {
    AssetInfo assetInfo;
    MeshMetaData meshMetaData;
  };

  /**
   * node: drawable's scene node
   *
   * meshID:
   * -) for non-ptex mesh:
   * meshID is the global index into meshes_.
   * meshes_[meshID] is the BaseMesh corresponding to the drawable;
   *
   * -) for ptex mesh:
   * meshID is the index of the submesh corresponding to the drawable;
   */
  struct StaticDrawableInfo {
    esp::scene::SceneNode& node;
    uint32_t meshID;
  };

  /**
   * @brief Define a map type referencing function pointers to @ref
   * createPrimitiveAttributes() keyed by string names of classes being
   * instanced, as defined in @ref PrimitiveNames3D
   */
  typedef std::map<std::string,
                   std::shared_ptr<Attrs::AbstractPrimitiveAttributes> (
                       esp::assets::ResourceManager::*)()>
      Map_Of_PrimTypes;

  //======== Scene Functions ========

  /**
   * @brief Recursive contruction of scene nodes for an asset.
   *
   * Creates a drawable for the component of an asset referenced by the @ref
   * MeshTransformNode and adds it to the @ref DrawableGroup as child of
   * parent.
   * @param metaData The @ref MeshMetaData object containing information about
   * the meshes, textures, materials, and component heirarchy of the asset.
   * @param parent The @ref scene::SceneNode of which the component will be a
   * child.
   * @param lightSetup The @ref LightSetup key that will be used
   * for the added component.
   * @param drawables The @ref DrawableGroup with which the component will be
   * rendered.
   * @param meshTransformNode The @ref MeshTransformNode for component
   * identifying its mesh, material, transformation, and children.
   * @param[out] visNodeCache Cache for pointers to all nodes created as the
   * result of this recursive process.
   * @param computeAABBs whether absolute bounding boxes should be computed
   * @param staticDrawableInfo structure holding the drawable infos for aabbs
   */
  void addComponent(const MeshMetaData& metaData,
                    scene::SceneNode& parent,
                    const Mn::ResourceKey& lightSetup,
                    DrawableGroup* drawables,
                    const MeshTransformNode& meshTransformNode,
                    std::vector<scene::SceneNode*>& visNodeCache,
                    bool computeAbsoluteAABBs,
                    std::vector<StaticDrawableInfo>& staticDrawableInfo);

  /**
   * @brief Load textures from importer into assets, and update metaData for
   * an asset to link textures to that asset.
   *
   * @param importer The importer already loaded with information for the
   * asset.
   * @param loadedAssetData The asset's @ref LoadedAssetData object.
   */
  void loadTextures(Importer& importer, LoadedAssetData& loadedAssetData);

  /**
   * @brief Load meshes from importer into assets.
   *
   * Compute bounding boxes, upload mesh data to GPU, and update metaData for
   * an asset to link meshes to that asset.
   * @param importer The importer already loaded with information for the
   * asset.
   * @param loadedAssetData The asset's @ref LoadedAssetData object.
   */
  void loadMeshes(Importer& importer, LoadedAssetData& loadedAssetData);

  /**
   * @brief Recursively parse the mesh component transformation heirarchy for
   * the imported asset.
   *
   * @param importer The importer already loaded with information for the
   * asset.
   * @param parent The root of the mesh transform heirarchy for the remaining
   * sub-tree. The generated @ref MeshTransformNode will be added as a child.
   * Typically the @ref MeshMetaData::root to begin recursion.
   * @param componentID The next component to add to the heirarchy. Identifies
   * the component in the @ref Importer.
   */
  void loadMeshHierarchy(Importer& importer,
                         MeshTransformNode& parent,
                         int componentID);

  /**
   * @brief Recursively build a unified @ref MeshData from loaded assets via a
   * tree of @ref MeshTransformNode.
   *
   * @param mesh The @ref MeshData being constructed.
   * @param metaData The @ref MeshMetaData for the object heirarchy being
   * joined.
   * @param node The current @ref MeshTransformNode in the recursion.
   * @param transformFromParentToWorld The cumulative transformation up to but
   * not including the current @ref MeshTransformNode.
   */
  void joinHeirarchy(MeshData& mesh,
                     const MeshMetaData& metaData,
                     const MeshTransformNode& node,
                     const Mn::Matrix4& transformFromParentToWorld);

  /**
   * @brief Load materials from importer into assets, and update metaData for
   * an asset to link materials to that asset.
   *
   * @param importer The importer already loaded with information for the
   * asset.
   * @param loadedAssetData The asset's @ref LoadedAssetData object.
   */
  void loadMaterials(Importer& importer, LoadedAssetData& loadedAssetData);

  /**
   * @brief Build a @ref PhongMaterialData for use with flat shading
   *
   * Textures must already be loaded for the asset this material belongs to
   *
   * @param material Material data with texture IDs
   * @param textureBaseIndex Base index of the assets textures in textures_
   */
  gfx::PhongMaterialData::uptr buildFlatShadedMaterialData(
      const Mn::Trade::PhongMaterialData& material,
      int textureBaseIndex);

  /**
   * @brief Build a @ref PhongMaterialData for use with phong shading
   *
   * Textures must already be loaded for the asset this material belongs to
   *
   * @param material Material data with texture IDs
   * @param textureBaseIndex Base index of the assets textures in textures_

   */
  gfx::PhongMaterialData::uptr buildPhongShadedMaterialData(
      const Mn::Trade::PhongMaterialData& material,
      int textureBaseIndex);

  /**
   * @brief Load a mesh describing some scene asset based on the passed
   * assetInfo.
   *
   * If both parent and drawables are provided, add the mesh to the
   * scene graph for rendering.
   * @param info The @ref AssetInfo for the mesh, already parsed from a file.
   * @param parent The @ref scene::SceneNode to which the mesh will be added
   * as a child.
   * @param drawables The @ref DrawableGroup with which the mesh will be
   * rendered.
   * @param computeAbsoluteAABBs Whether absolute bounding boxes should be
   * computed
   * @param splitSemanticMesh Split the semantic mesh by objectID, used for A/B
   * testing
   * @param lightSetup The @ref LightSetup key that will be used
   * for the loaded asset.
   */
  bool loadStageInternal(
      const AssetInfo& info,
      std::shared_ptr<physics::PhysicsManager> _physicsManager,
      scene::SceneNode* parent = nullptr,
      DrawableGroup* drawables = nullptr,
      bool computeAbsoluteAABBs = false,
      bool splitSemanticMesh = true,
      const Mn::ResourceKey& lightSetup = Mn::ResourceKey{NO_LIGHT_KEY});

  /**
   * @brief Creates a map of appropriate asset infos for sceneries.  Will always
   * create render asset info.  Will create collision asset info and semantic
   * stage asset info if requested.
   *
   * @param stageAttributes The stage attributes file holding the stage's
   * information.
   * @param createCollisionInfo Whether collision-based asset info should be
   * created (only if physicsManager type is not none)
   * @param createSemanticInfo Whether semantic mesh-based asset info should be
   * created
   */
  std::map<std::string, AssetInfo> createStageAssetInfosFromAttributes(
      const Attrs::StageAttributes::ptr& stageAttributes,
      bool createCollisionInfo,
      bool createSemanticInfo);

  /**
   * @brief Load a PTex mesh into assets from a file and add it to the scene
   * graph for rendering.
   * @return true if the mesh is loaded, otherwise false
   *
   * @param info The @ref AssetInfo for the mesh, already parsed from a
   * file.
   * @param parent The @ref scene::SceneNode to which the mesh will be added
   * as a child.
   * @param drawables The @ref DrawableGroup with which the mesh will be
   * rendered.
   */
  bool loadPTexMeshData(const AssetInfo& info,
                        scene::SceneNode* parent,
                        DrawableGroup* drawables);

  /**
   * @brief Load an instance mesh (e.g. Matterport reconstruction) into assets
   * from a file and add it to the scene graph for rendering.
   *
   * @param info The @ref AssetInfo for the mesh, already parsed from a file.
   * @param parent The @ref scene::SceneNode to which the mesh will be added
   * as a child.
   * @param drawables The @ref DrawableGroup with which the mesh will be
   * rendered.
   * @param computeAbsoluteAABBs Whether absolute bounding boxes should be
   * computed
   * @param splitSemanticMesh Split the semantic mesh by objectID
   */
  bool loadInstanceMeshData(const AssetInfo& info,
                            scene::SceneNode* parent,
                            DrawableGroup* drawables,
                            bool computeAbsoluteAABBs,
                            bool splitSemanticMesh);  // was default true

  /**
   * @brief Load a mesh (e.g. gltf) into assets from a file.
   *
   * If both parent and drawables are provided, add the mesh to the
   * scene graph for rendering.
   * @param info The @ref AssetInfo for the mesh, already parsed from a file.
   * @param parent The @ref scene::SceneNode to which the mesh will be added
   * as a child.
   * @param drawables The @ref DrawableGroup with which the mesh will be
   * rendered.
   * @param computeAbsoluteAABBs Whether absolute bounding boxes should be
   * computed
   * @param lightSetup The @ref LightSetup key that will be used
   * for the loaded asset.
   */
  bool loadGeneralMeshData(const AssetInfo& info,
                           scene::SceneNode* parent = nullptr,
                           DrawableGroup* drawables = nullptr,
                           bool computeAbsoluteAABBs = false,
                           const Mn::ResourceKey& lightSetup = Mn::ResourceKey{
                               NO_LIGHT_KEY});

  /**
   * @brief Load a SUNCG mesh into assets from a file. !Deprecated! TODO:
   * remove?
   *
   * @param info The @ref AssetInfo for the mesh, already parsed from a file.
   * @param parent The @ref scene::SceneNode to which the mesh will be added
   * as a child.
   * @param drawables The @ref DrawableGroup with which the mesh will be
   * rendered.
   */
  bool loadSUNCGHouseFile(const AssetInfo& info,
                          scene::SceneNode* parent,
                          DrawableGroup* drawables);

  /**
   * @brief initialize default lighting setups in the current ShaderManager
   */
  void initDefaultLightSetups();

  /**
   * @brief initialize default material setups in the current ShaderManager
   */
  void initDefaultMaterials();

  /**
   * @brief Checks if light setup is compatible with loaded asset
   */
  bool isLightSetupCompatible(const LoadedAssetData& loadedAssetData,
                              const Mn::ResourceKey& lightSetup) const;

  // ======== Geometry helper functions, data structures ========

  /**
   * @brief Apply a translation to the vertices of a mesh asset and store that
   * transformation in @ref BaseMesh::meshTransform_.
   *
   * @param meshDataGL The mesh data.
   * @param translation The translation transform to apply.
   */
  void translateMesh(BaseMesh* meshDataGL, Mn::Vector3 translation);

  /**
   * @brief Compute and return the axis aligned bounding box of a mesh in mesh
   * local space
   * @param meshDataGL The mesh data.
   * @return The mesh bounding box.
   */
  Mn::Range3D computeMeshBB(BaseMesh* meshDataGL);

  /**
   * @brief Compute the absolute AABBs for drawables in PTex mesh in world
   * space
   * @param baseMesh: ptex mesh
   */
#ifdef ESP_BUILD_PTEX_SUPPORT
  void computePTexMeshAbsoluteAABBs(
      BaseMesh& baseMesh,
      const std::vector<StaticDrawableInfo>& staticDrawableInfo);
#endif

  /**
   * @brief Compute the absolute AABBs for drawables in general mesh (e.g.,
   * MP3D) world space
   */
  void computeGeneralMeshAbsoluteAABBs(
      const std::vector<StaticDrawableInfo>& staticDrawableInfo);

  /**
   * @brief Compute the absolute AABBs for drawables in semantic mesh in world
   * space
   */
  void computeInstanceMeshAbsoluteAABBs(
      const std::vector<StaticDrawableInfo>& staticDrawableInfo);

  /**
   * @brief Compute absolute transformations of all drwables stored in
   * staticDrawableInfo_
   */
  std::vector<Mn::Matrix4> computeAbsoluteTransformations(
      const std::vector<StaticDrawableInfo>& staticDrawableInfo);

  // ======== Rendering Utility Functions ========

  /**
   * @brief Create a @ref gfx::Drawable for the specified mesh, node,
   * and @ref ShaderType.
   *
   * Add this drawable to the @ref DrawableGroup if provided.
   * @param shaderType Indentifies the desired shader program for rendering
   * the
   * @ref gfx::Drawable.
   * @param mesh The render mesh.
   * @param node The @ref scene::SceneNode to which the drawable will be
   * attached.
   * @param lightSetup The @ref LightSetup key that will be used
   * for the drawable.
   * @param material The @ref MaterialData key that will be used
   * for the drawable.
   * @param meshID Optional, the index of this mesh component stored in
   * meshes_
   * @param group Optional @ref DrawableGroup with which the render the @ref
   * gfx::Drawable.
   * @param texture Optional texture for the mesh.
   * @param color Optional color parameter for the shader program. Defaults to
   * white.
   */
  void createGenericDrawable(Mn::GL::Mesh& mesh,
                             scene::SceneNode& node,
                             const Mn::ResourceKey& lightSetup,
                             const Mn::ResourceKey& material,
                             DrawableGroup* group = nullptr);

  // ======== General geometry data ========
  // shared_ptr is used here, instead of Corrade::Containers::Optional, or
  // std::optional because shared_ptr is reference type, not value type, and
  // thus we can avoiding duplicated loading

  /**
   * @brief The mesh data for loaded assets.
   */
  std::vector<std::shared_ptr<BaseMesh>> meshes_;

  /**
   * @brief The texture data for loaded assets.
   */
  std::vector<std::shared_ptr<Mn::GL::Texture2D>> textures_;

  /**
   * @brief The next available unique ID for loaded materials
   */
  int nextMaterialID_ = 0;

  /**
   * @brief Asset metadata linking meshes, textures, materials, and the
   * component transformation heirarchy for loaded assets.
   *
   * Maps absolute path keys to metadata.
   */
  std::map<std::string, LoadedAssetData> resourceDict_;

  /**
   * @brief The @ref ShaderManager used to store shader information for
   * drawables created by this ResourceManager
   */
  gfx::ShaderManager shaderManager_;

  // ======== File and primitive importers ========
  /**
   * @brief Plugin Manager used to instantiate importers which in turn are used
   * to load asset data
   */
  Corrade::PluginManager::Manager<Importer> importerManager_;

  /**
   * @brief Importer used to synthesize Magnum Primitives (PrimitiveImporter).
   * This object allows for similar usage to File-based importers, but requires
   * no file to be available/read.
   */
  Corrade::Containers::Pointer<Importer> primitiveImporter_;

  /**
   * @brief Importer used to load generic mesh files (AnySceneImporter)
   */
  Corrade::Containers::Pointer<Importer> fileImporter_;

  // ======== Physical parameter data ========

  /**
   * @brief Manages all construction and access to asset attributes.
   */
  managers::AssetAttributesManager::ptr assetAttributesManager_ = nullptr;

  /**
   * @brief Manages all construction and access to object attributes.
   */
  managers::ObjectAttributesManager::ptr objectAttributesManager_ = nullptr;

  /**
   * @brief Manages all construction and access to physics world attributes.
   */
  managers::PhysicsAttributesManager::ptr physicsAttributesManager_ = nullptr;

  /**
   * @brief Manages all construction and access to scene attributes.
   */
  managers::StageAttributesManager::ptr stageAttributesManager_ = nullptr;

  //! tracks primitive mesh ids
  int nextPrimitiveMeshId = 0;
  /**
   * @brief Primitive meshes available for instancing via @ref
   * addPrimitiveToDrawables for debugging or visualization purposes.
   */
  std::map<int, std::unique_ptr<Mn::GL::Mesh>> primitive_meshes_;

  /**
   * @brief Maps string keys (typically property filenames) to @ref
   * CollisionMeshData for all components of a loaded asset.
   */
  std::map<std::string, std::vector<CollisionMeshData>> collisionMeshGroups_;

  /**
   * @brief Flag to denote the desire to compress textures. TODO: unused?
   */
  bool compressTextures_ = false;
};

}  // namespace assets
}  // namespace esp

#endif  // ESP_ASSETS_RESOURCEMANAGER_H_
